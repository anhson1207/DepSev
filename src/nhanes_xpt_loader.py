"""Read audited NHANES XPORT files without changing their original bytes.

Pandas can decode an all-zero IBM numeric word as 2**-260. Only independently
verified zero words in previously audited variables are corrected. Small values
with nonzero source bytes, SAS missing values, and unaudited files are not guessed.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nhanes_official_utils import ARTIFACTS, ROOT, TABLES, sha256

IBM_ZERO_ARTIFACT = 2.0 ** -260


def normalize_verified_zeros(values, zero_mask):
    """Return a copy and count; the caller must verify the source-byte mask."""
    mask = np.asarray(zero_mask, dtype=bool)
    if mask.shape != (len(values),):
        raise ValueError("Zero mask does not match the decoded records")
    if not values.loc[mask].isin([0.0, IBM_ZERO_ARTIFACT]).all():
        raise ValueError("Unexpected decoding of verified zero words; re-audit reader")
    corrected = mask & values.eq(IBM_ZERO_ARTIFACT).to_numpy()
    result = values.copy()
    result.loc[corrected] = 0.0
    return result, int(corrected.sum())


def load_nhanes_xpt(path, columns):
    """Return requested columns and a per-variable zero-normalization audit.

    This processing loader intentionally requires the existing official-file
    ledger and reader audit. It refuses unaudited variables or changed raw files.
    Column names are normalized to uppercase, not questionnaire values.
    """
    path = Path(path).resolve()
    key = str(path.relative_to(ROOT))
    ledger = json.loads((ARTIFACTS / "nhanes_official_file_ledger.json").read_text())
    if key not in ledger or ledger[key]["kind"] != "xpt":
        raise ValueError(f"File is not an audited official XPT: {key}")
    digest = sha256(path)
    if digest != ledger[key]["sha256"]:
        raise ValueError(f"Raw hash differs from download audit: {key}")
    cycle, module = path.parent.parent.name, path.stem
    previous = pd.read_csv(TABLES / "nhanes_official_reader_audit.csv")
    previous = previous.loc[previous.cycle.eq(cycle) & previous.module.eq(module)]
    previous = previous.set_index("variable")
    if not previous.index.is_unique:
        raise ValueError("Previous reader audit has duplicated variable entries")
    columns = list(dict.fromkeys(column.upper() for column in columns))
    if not set(columns).issubset(previous.index):
        raise ValueError(f"Requested variables have no reader audit: {columns}")

    with pd.read_sas(path, format="xport", iterator=True) as reader:
        fields = {field["name"].decode().upper(): field for field in reader.fields}
        start, width, nobs = reader.record_start, reader.record_length, reader.nobs
        frame = reader.read()
    frame.columns = frame.columns.str.upper()
    if len(frame) != nobs:
        raise ValueError("XPORT metadata and decoded row counts disagree")
    payload = path.read_bytes()
    raw = np.frombuffer(payload, dtype=np.uint8, count=nobs * width,
                        offset=start).reshape(nobs, width)
    result = frame[columns].copy()
    audits = []
    for column in columns:
        field = fields[column]
        if field["ntype"] != "numeric":
            raise ValueError(f"Unexpected nonnumeric audited variable: {column}")
        position, length = field["npos"], field["field_length"]
        if not 2 <= length <= 8 or position + length > width:
            raise ValueError("Unexpected XPORT numeric field layout")
        zero = (raw[:, position:position + length] == 0).all(axis=1)
        old = previous.loc[column]
        if int(zero.sum()) != int(old.raw_zero_byte_count):
            raise ValueError(f"Raw zero count changed since reader audit: {column}")
        result[column], count = normalize_verified_zeros(result[column], zero)
        expected = old.codebook_zero_count_if_explicit
        if column.startswith("DPQ") and pd.notna(expected) and zero.sum() != int(expected):
            raise ValueError(f"PHQ zero count differs from official codebook: {column}")
        audits.append({
            "cycle": cycle, "file": key, "variable": column,
            "n_records": nobs, "raw_zero_byte_count": int(zero.sum()),
            "n_cells_corrected": count,
            "zero_count_after": int(result[column].eq(0).sum()),
            "codebook_zero_count_if_explicit": expected,
            "pandas_version": pd.__version__, "raw_sha256": digest,
            "raw_hash_unchanged": True,
            "condition": "all source numeric bytes zero AND decoded value exactly 2**-260",
        })
    if sha256(path) != digest:
        raise ValueError("Raw file changed while reading")
    return result, audits
