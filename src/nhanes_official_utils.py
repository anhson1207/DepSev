"""Small, read-only-data helpers for the official NHANES download/audit stage."""

import csv
import hashlib
import json
import socket
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw_official"
ARTIFACTS = ROOT / "artifacts"
TABLES = ROOT / "results/tables"
CYCLES = ("2011_2012", "2013_2014", "2015_2016")
LEDGER = ARTIFACTS / "nhanes_official_file_ledger.json"
PRESERVATION = ARTIFACTS / "nhanes_official_preservation.json"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing an empty audit table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def preserve_existing_files():
    """Snapshot pre-existing research files, then require byte-for-byte preservation."""
    if PRESERVATION.exists():
        check_preservation()
        return
    paths = []
    for directory in ("data/raw", "data/processed", "artifacts", "results", "src"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.name != ".DS_Store")
    paths.extend(ROOT / name for name in ("README.md", "requirements.txt", ".gitignore"))
    new_names = {"11_download_nhanes_official.py", "12_audit_nhanes_official.py",
                 "nhanes_official_utils.py"}
    preserved = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(paths))
                 if p.exists() and p.name not in new_names}
    save_json(PRESERVATION, {"created_at": utc_now(), "sha256": preserved})
    print(f"Preservation snapshot: {len(preserved)} existing files", flush=True)


def check_preservation():
    saved = json.loads(PRESERVATION.read_text())["sha256"]
    changed = [name for name, digest in saved.items()
               if not (ROOT / name).is_file() or sha256(ROOT / name) != digest]
    if changed:
        raise ValueError(f"Pre-existing files changed: {changed}")
    return len(saved)


def official_url(url):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in {"wwwn.cdc.gov", "www.cdc.gov", "cdc.gov"}:
        raise ValueError(f"Only official CDC HTTPS URLs are allowed: {url}")
    return url


class OfficialRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        official_url(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


class Downloader:
    """Store original response bytes; skip only if a previously recorded hash matches.

    The local SHA-256 establishes reproducibility, not a CDC-published checksum.
    Untracked files and hash mismatches stop the run instead of being overwritten.
    """

    def __init__(self):
        self.ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
        self.opener = build_opener(OfficialRedirects())

    def fetch(self, url, path, kind):
        official_url(url)
        key = str(path.relative_to(ROOT))
        if path.exists():
            record = self.ledger.get(key)
            if not record or record["url"] != url or sha256(path) != record["sha256"]:
                raise ValueError(f"Untracked/changed file; will not overwrite: {key}")
            print(f"SKIP verified {key}", flush=True)
            return record
        request = Request(url, headers={"User-Agent": "DepSev-ML research data audit"})
        for attempt in range(3):
            try:
                with self.opener.open(request, timeout=45) as response:
                    official_url(response.url)
                    payload = response.read()
                    status = response.status
                    final_url = response.url
                    expected = response.headers.get("Content-Length")
                if status != 200 or not payload or (expected and len(payload) != int(expected)):
                    raise ValueError(f"Incomplete HTTP response for {url}")
                if kind == "xpt" and not payload.startswith(b"HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!"):
                    raise ValueError(f"Response is not a SAS XPORT file: {url}")
                if kind == "html" and b"<html" not in payload[:5000].lower():
                    raise ValueError(f"Response is not HTML: {url}")
                if kind == "pdf" and not payload.startswith(b"%PDF-"):
                    raise ValueError(f"Response is not PDF: {url}")
                break
            except (HTTPError, URLError, TimeoutError, socket.timeout) as error:
                if isinstance(error, HTTPError) and error.code not in {429, 500, 502, 503, 504}:
                    raise RuntimeError(f"Official source unavailable; no fallback: {url}") from error
                if attempt == 2:
                    raise RuntimeError(f"Official network failed after 3 attempts: {url}") from error
                print(f"RETRY {attempt + 1}/2 {url}: {error}", flush=True)
                time.sleep(2 ** (attempt + 1))
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents overwriting any existing raw file.
        with path.open("xb") as stream:
            stream.write(payload)
        record = {"url": url, "resolved_url": final_url, "local_path": key,
                  "kind": kind, "http_status": status, "size_bytes": len(payload),
                  "sha256": sha256(path), "downloaded_at": utc_now()}
        self.ledger[key] = record
        save_json(LEDGER, self.ledger)
        print(f"DOWNLOADED {key}: {len(payload):,} bytes", flush=True)
        return record

    def verify_all(self):
        for name, record in self.ledger.items():
            if sha256(ROOT / name) != record["sha256"]:
                raise ValueError(f"Downloaded bytes changed: {name}")
        return len(self.ledger)


class Node:
    def __init__(self, tag="root", attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []

    def walk(self, tag=None):
        if tag is None or self.tag == tag:
            yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.walk(tag)

    def text(self):
        return " ".join(" ".join(child.text() if isinstance(child, Node) else child
                                 for child in self.children).split())


class Document(HTMLParser):
    """Enough DOM structure to inspect official tables without extra dependencies."""
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "wbr"}

    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.root = Node()
        self.stack = [self.root]
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in self.VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(Node(tag, attrs))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def table_rows(node):
    return [[cell.text() for cell in row.children
             if isinstance(cell, Node) and cell.tag in {"td", "th"}]
            for row in node.walk("tr")]


def absolute_link(base, href):
    return official_url(urljoin(base, href))
