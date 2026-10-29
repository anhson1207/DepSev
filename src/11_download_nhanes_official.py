"""Discover, download and verify official NHANES files; never build/fit a dataset.

Run with the project environment: python src/11_download_nhanes_official.py
Use --discover-only to save the CDC-verified catalog before downloading XPTs.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from config import FEATURE_SPECS, PHQ9_COLUMNS
from nhanes_official_utils import (
    ARTIFACTS, CYCLES, RAW, ROOT, TABLES, Document, Downloader, Node,
    absolute_link, check_preservation, preserve_existing_files, save_csv,
    save_json, sha256, table_rows, utc_now,
)

CATALOG = ARTIFACTS / "nhanes_official_source_catalog.json"
DEPENDENCIES = ARTIFACTS / "nhanes_official_feature_dependencies.json"
MANIFEST = ARTIFACTS / "nhanes_official_download_manifest.csv"


def feature_dependencies():
    """Use reviewed source definitions; do not infer derived names from suffixes."""
    schema = json.loads((ARTIFACTS / "final_feature_schema.json").read_text())
    inventory = pd.read_csv(TABLES / "ablation_feature_inventory.csv")
    assert inventory.feature.tolist() == schema["ordered_features"]
    assert len(inventory) == 34
    config = {row["feature"]: row for row in FEATURE_SPECS}
    rows = []
    for row in inventory.to_dict("records"):
        feature = row["feature"]
        sources = row["source_columns"].split(";")
        if feature in {"SBP_mean", "DBP_mean"}:
            sources = config[feature]["source_columns"]
        rows.append({"feature": feature, "group": row["group"],
                     "source_variables": sources, "source_file": row["source_file"],
                     "derived": sources != [feature], "description": row["description"]})
    # Explicit assertions protect the dependency review from silent code drift.
    expected = {
        "SBP_mean": ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"],
        "DBP_mean": ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"],
        "smoking_status": ["SMQ020", "SMQ040"],
        "alcohol_status": ["ALQ101", "ALQ110", "ALQ120Q", "ALQ130"],
        "alcohol_quantity_code": ["ALQ110", "ALQ120Q", "ALQ130"],
    }
    assert {r["feature"]: r["source_variables"] for r in rows if r["derived"]} == expected
    sources = ["src/config.py", "src/ablation_features.py", "src/02_build_cohort.py",
               "src/05_feature_ablation.py", "artifacts/final_model_spec.json",
               "artifacts/final_feature_schema.json", "results/tables/ablation_feature_inventory.csv"]
    payload = {"features": rows, "phq9_items": PHQ9_COLUMNS,
               "source_sha256": {p: sha256(ROOT / p) for p in sources},
               "derived_rule_notes": {
                   "SBP_mean": "Mean of available readings 1-4, skipna=True; no exclusion of first reading.",
                   "DBP_mean": "Mean of available readings 1-4, skipna=True; no removal of zero readings.",
                   "smoking_status": "SMQ020 lifetime 100-cigarette gate, then SMQ040 current frequency.",
                   "alcohol_status": "ALQ101 context, ALQ110 lifetime gate, ALQ120Q past-year frequency, ALQ130 quantity.",
                   "alcohol_quantity_code": "ALQ130; tagged zero only for ALQ110=2 or ALQ120Q=0; not zero lifetime consumption.",
               }}
    if DEPENDENCIES.exists() and json.loads(DEPENDENCIES.read_text()) != payload:
        raise ValueError("Reviewed dependency definitions changed; human review required")
    save_json(DEPENDENCIES, payload)
    return payload


def discover(downloader, dependencies):
    required = set(PHQ9_COLUMNS)
    components = {"demographic.csv": "Demographics", "examination.csv": "Examination",
                  "labs.csv": "Laboratory", "questionnaire.csv": "Questionnaire"}
    expected_component = {v: "Questionnaire" for v in PHQ9_COLUMNS}
    for feature in dependencies["features"]:
        required.update(feature["source_variables"])
        expected_component.update({v: components[feature["source_file"]]
                                   for v in feature["source_variables"]})
    catalog = {"discovered_at": utc_now(), "required_raw_variables": sorted(required), "cycles": {}}
    for cycle in CYCLES:
        label = cycle.replace("_", "-")
        base = "https://wwwn.cdc.gov/nchs/nhanes/search/"
        variable_url = base + f"variablelist.aspx?Cycle={label}"
        data_url = base + f"DataPage.aspx?Cycle={label}"
        paths = {"variables": RAW / cycle / "docs/_variable_index.html",
                 "data": RAW / cycle / "docs/_data_index.html"}
        downloader.fetch(variable_url, paths["variables"], "html")
        downloader.fetch(data_url, paths["data"], "html")
        variables = []
        for cells in table_rows(Document(paths["variables"].read_text()).root):
            if len(cells) == 8 and cells[0] not in {"Variable Name", "VariableName"}:
                variables.append(dict(zip(("variable", "description", "module", "module_name",
                                           "begin_year", "end_year", "component", "constraints"), cells)))
        mapping = {}
        for name in sorted(required):
            # Restricted youth modules can repeat public adult DPQ variables.
            # CDC's index/codebook uses e.g. MCQ160b; SAS XPT uses MCQ160B.
            # This identifies the module only; actual XPT names are verified later.
            matches = [r for r in variables if r["variable"].upper() == name.upper()
                       and r["constraints"] == "None"
                       and r["component"] == expected_component[name]]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous module for {cycle}/{name}: {matches}")
            if matches:
                mapping[name] = matches[0]
        missing = sorted(required - set(mapping))
        if set(PHQ9_COLUMNS) & set(missing):
            raise ValueError(f"STOP compatibility for {cycle}: missing PHQ items {missing}")
        selected = {row["module"]: row for row in mapping.values()}
        # Inspect the official Sleep Disorders file even if SLD010H is absent.
        # This does NOT substitute another variable for the frozen feature.
        for row in variables:
            if row["module_name"] == "Sleep Disorders":
                selected.setdefault(row["module"], row)
        file_rows = {}
        for row in Document(paths["data"].read_text()).root.walk("tr"):
            links = [a.attrs.get("href", "") for a in row.walk("a")]
            xpts = [x for x in links if x.lower().endswith(".xpt")]
            docs = [x for x in links if x.lower().endswith((".htm", ".html"))]
            for href in xpts:
                module = Path(href).stem
                matched = [x for x in docs if Path(x).stem == module]
                if module in selected and len(matched) == 1:
                    file_rows[module] = {"cycle": cycle, "component": selected[module]["component"],
                                         "module": module, "module_name": selected[module]["module_name"],
                                         "filename": Path(href).name,
                                         "xpt_url": absolute_link(data_url, href),
                                         "documentation_url": absolute_link(data_url, matched[0]),
                                         "verified_on_data_page": data_url,
                                         "verified_on_variable_page": variable_url}
        if set(selected) != set(file_rows):
            raise ValueError(f"Module URLs not verified on CDC page: {set(selected) - set(file_rows)}")
        catalog["cycles"][cycle] = {"modules": [file_rows[k] for k in sorted(file_rows)],
                                     "variable_mapping": mapping, "missing_variables": missing}
        print(f"DISCOVERED {cycle}: {len(file_rows)} verified modules; missing={missing}", flush=True)
    save_json(CATALOG, catalog)
    return catalog


def download_lab_methods(downloader):
    """Follow verified CDC method-index links, selecting only the five F4 analytes."""
    manifest = []
    for cycle in CYCLES:
        index = RAW / cycle / "docs/_data_index.html"
        base = "https://wwwn.cdc.gov"
        candidates = {a.attrs.get("href", "") for a in Document(index.read_text()).root.walk("a")
                      if "labmethods.aspx?Cycle=" + cycle.replace("_", "-") in a.attrs.get("href", "")}
        urls = {absolute_link(base, href) for href in candidates}
        if len(urls) != 1:
            raise ValueError(f"Cannot verify laboratory method index link: {cycle}")
        url = urls.pop()
        path = RAW / cycle / "docs/_laboratory_methods.html"
        downloader.fetch(url, path, "html")
        for row in Document(path.read_text()).root.walk("tr"):
            cells = [n.text() for n in row.children if isinstance(n, Node) and n.tag == "td"]
            if not cells:
                continue
            label = cells[0].strip()
            wanted = label in {"Albumin", "Creatinine"}
            wanted = wanted or label.startswith(("Albumin (DXC", "Creatinine (DXC",
                                                "Total Cholesterol", "HDL- Cholesterol",
                                                "HDL - Cholesterol", "Glycohemoglobin"))
            if not wanted:
                continue
            for link in row.walk("a"):
                href = link.attrs.get("href", "")
                if not href.lower().endswith(".pdf"):
                    continue
                pdf_url = absolute_link(url, href)
                target = RAW / cycle / "docs" / Path(href).name
                record = downloader.fetch(pdf_url, target, "pdf")
                manifest.append({"cycle": cycle, "analyte": label,
                                 "verified_on": url, **record})
    save_csv(ARTIFACTS / "nhanes_official_lab_method_manifest.csv", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--methods-only", action="store_true")
    args = parser.parse_args()
    preserve_existing_files()
    dependencies = feature_dependencies()
    downloader = Downloader()
    if args.methods_only:
        download_lab_methods(downloader)
        return
    catalog = discover(downloader, dependencies)
    if args.discover_only:
        return
    manifest, verification = [], []
    modules = [row for c in CYCLES for row in catalog["cycles"][c]["modules"]]
    for index, row in enumerate(modules, 1):
        cycle = row["cycle"]
        print(f"[{index}/{len(modules)}] {cycle}/{row['module']}", flush=True)
        xpt = RAW / cycle / "xpt" / row["filename"]
        doc = RAW / cycle / "docs" / Path(row["documentation_url"]).name
        documentation = downloader.fetch(row["documentation_url"], doc, "html")
        record = downloader.fetch(row["xpt_url"], xpt, "xpt")
        frame = pd.read_sas(xpt, format="xport")
        has_id = "SEQN" in frame
        check = {"cycle": cycle, "filename": xpt.name, "n_rows": len(frame),
                 "n_columns": len(frame.columns), "seqn_present": has_id,
                 "unique_seqn": int(frame.SEQN.nunique()) if has_id else None,
                 "duplicate_seqn": int(frame.SEQN.duplicated().sum()) if has_id else None,
                 "missing_seqn": int(frame.SEQN.isna().sum()) if has_id else None}
        print(f"VERIFY {check}", flush=True)
        if not has_id or check["duplicate_seqn"] or check["missing_seqn"]:
            raise ValueError(f"Invalid participant identifiers: {check}")
        if row["module_name"] == "Mental Health - Depression Screener" and not set(PHQ9_COLUMNS) <= set(frame):
            raise ValueError(f"STOP compatibility for {cycle}: PHQ-9 incomplete in XPT")
        verification.append(check)
        manifest.append({**row, **{k: record[k] for k in ("local_path", "http_status", "size_bytes", "sha256", "downloaded_at")},
                         "documentation_local_path": documentation["local_path"],
                         "documentation_sha256": documentation["sha256"]})
        save_csv(MANIFEST, manifest)
        save_csv(TABLES / "nhanes_official_xpt_verification.csv", verification)
    download_lab_methods(downloader)
    print(f"Verified {downloader.verify_all()} downloads; {check_preservation()} old files unchanged.")


if __name__ == "__main__":
    main()
