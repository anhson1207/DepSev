"""Audit raw variables/codebooks and 2013 parity without harmonizing or fitting.

No cohort joins, derived-feature construction, PHQ scoring, or model calls occur.
"""

import argparse
import json
import re

import numpy as np
import pandas as pd

from nhanes_official_utils import (
    ARTIFACTS, CYCLES, RAW, ROOT, TABLES, Document, Downloader, Node,
    check_preservation, save_csv, save_json, sha256, table_rows, utc_now,
)

EVIDENCE = ARTIFACTS / "nhanes_official_codebook_evidence.json"

# Semantic review of the downloaded codebooks, not a column-name-only match.
# EXACT means the same measured construct, unit and value coding in age >=20.
# It does not claim identical wording, instruments, distributions or lab sites.
FEATURE_NOTES = {
    "BMXBMI": "Same weight/height-squared definition, kg/m^2, rounded to 1 decimal. 2011 HTML omits frequency tables, not the variable or measurement definition.",
    "BMXWAIST": "Same waist circumference in cm. 2011 HTML omits frequency tables. No adult exclusions for this measure in the three component descriptions.",
    "BPXPLS": "Same published 60-second pulse (30-second count times 2). 2015 component text lowers pulse eligibility to age 5, while its variable Target still says 8. No age>=20 coverage difference; retain documentation discrepancy.",
    "SBP_mean": "Same four raw systolic readings, mmHg. Preserve current mean of available readings 1-4, including the first reading. No feature is constructed in this audit.",
    "DBP_mean": "Same four raw diastolic readings, mmHg; zero is explicitly permitted by all codebooks. Preserve mean of available readings 1-4, without newly excluding zero or the first reading.",
    "LBXGH": "HbA1c % and adult eligibility unchanged. 2011 method PDFs explicitly document G7 to G8 within-cycle transition; 2013/2015 use G8. 2013 HTML also notes lab-site change. No unverified calibration applied; HTML alone incompletely describes 2011 equipment changes.",
    "LBXTC": "Same enzymatic total cholesterol in mg/dL, not BIOPRO LBXSCH. Roche Modular P in 2011; Modular P plus Cobas 6000 in 2013; 2015 reports no further change. 2011 numerical LLOD is not stated in the reviewed HTML/SOP; clinical reportable lower bound 10 is NOT substituted for LLOD. 2013/2015 HTML LLOD=4 mg/dL.",
    "LBDHDD": "Same direct HDL in mg/dL. Roche Modular P to Modular P/Cobas 6000 equipment transition in 2013. HTML LLOD tables call the analyte LBXHDD, but codebook and XPT variable are LBDHDD. 2011 SOP reports a clinical range of 3-200, not an explicit numerical LLOD; later HTML LLOD=3 mg/dL. No variable substitution.",
    "LBXSCR": "Same serum creatinine, mg/dL, IDMS-standardized Jaffe rate method. DxC800 in 2011/2013; DxC800 and DxC660i in 2015. Analytical lower bound/sensitivity 0.1 mg/dL in all method PDFs; later HTML LLOD=0.1. No conversion or calibration performed.",
    "LBXSAL": "Same serum albumin, g/dL, bromcresol-purple endpoint method. DxC800 in 2011/2013; DxC800 and DxC660i in 2015. Analytical lower bound/sensitivity 1 g/dL in all method PDFs; later HTML LLOD=1. No conversion performed.",
    "BPQ020": "Same ever clinician-told hypertension question and 1/2/7/9 coding; age>=16. Downstream skip destination changes BPQ057/BPQ056/BPQ080, not eligibility or meaning of BPQ020 itself.",
    "DIQ010": "Same clinician-told diabetes question, pregnancy wording and 1=yes/2=no/3=borderline/7/9 coding; preserve borderline as a valid category.",
    "MCQ160B": "Same clinician-told congestive heart failure question, age>=20 and 1/2/7/9. Codebook anchor MCQ160b; actual XPT name MCQ160B in all cycles.",
    "MCQ160C": "Same clinician-told coronary heart disease question, age>=20 and 1/2/7/9. Codebook anchor MCQ160c; actual XPT name MCQ160C.",
    "MCQ160D": "Same clinician-told angina question, age>=20 and 1/2/7/9. Codebook anchor MCQ160d; actual XPT name MCQ160D.",
    "MCQ160E": "Same clinician-told heart attack question, age>=20 and 1/2/7/9. Codebook anchor MCQ160e; actual XPT name MCQ160E.",
    "MCQ160F": "Same clinician-told stroke question, age>=20 and 1/2/7/9. Codebook anchor MCQ160f; actual XPT name MCQ160F.",
    "MCQ220": "Same clinician-told cancer/malignancy question, age>=20 and 1/2/7/9. Do not impute age-ineligible youth into the adult feature definition.",
    "HUQ010": "Same five ordered general-health responses and 7/9 nonresponse codes, all ages; no recoding of poor health into a depression target.",
    "PAQ605": "Conservative DIFFERENT_DEFINITION review flag: 2011 introduction explicitly includes studying/training in work and defines intensity; 2013/2015 omit these phrases. Same 10-minute vigorous-activity criterion and codes, but question scope is not verbatim equivalent. Cannot repair a wording/scope difference by a numerical recode; researcher decision required, not automatic exclusion.",
    "PAQ620": "Same moderate work activity >=10 minutes, 1/2/7/9, age>=12. Its own wording is identical, but it follows the changed PAQ605 work-introduction context; retain this contextual limitation.",
    "PAQ650": "Same vigorous recreation >=10 minutes and 1/2/7/9, age>=12. Later wording explicitly repeats typical week; 2011 establishes typical week in PAQ605 introduction. Transport/transportation wording differs; same operational construct.",
    "PAQ665": "Same moderate recreation >=10 minutes and 1/2/7/9, age>=12. Later wording repeats typical week and changes example golf to volleyball; retain wording difference, without treating an illustrative example as a new activity threshold.",
    "PAD680": "Conservative DIFFERENT_DEFINITION flag includes measurement-protocol differences: typical sitting time remains minutes/day, but work/school framing and low-time probes differ. 2011 notes a mid-cycle <8-hour probe; 2013 instructions include <8 hours; 2015 include <3 hours. Same 7777/9999 codes do not establish full measurement equivalence. No value recode can undo interview probing.",
    "smoking_status": "SMQ020>=100 lifetime cigarettes gates SMQ040 current daily/some-days/not-at-all. Public item eligibility age>=20 in 2011 vs >=18 later; compatible for the pre-existing age>=20 scope. 2015 adds separate under-100 questions, not used here. Preserve under_100_lifetime, not a claim of never smoking.",
    "ALQ101": "Same >=12 drinks in any one year, not necessarily the past year, 1/2/7/9, public age>=18. Introductory wording shortened in 2015. Later module docs explicitly state cognitive/language exclusions; 2011 is less explicit, so full exclusion equivalence is not established solely by its Target line.",
    "alcohol_status": "ALQ101/ALQ110/ALQ120Q/ALQ130 routing preserves under-12-lifetime, no-drinks-past-year, drank-past-year states. ALQ130 top-code change does not affect positive-vs-zero applicability. ALQ120Q is frequency quantity with units in ALQ120U, NOT inherently days/year; current rule uses only zero/positive, so ALQ120U is context, not an omitted numerical conversion dependency.",
    "alcohol_quantity_code": "ALQ130 instructions in 2011/2013 cap at 95; 2015 releases 15 as 15 drinks or more. Numeric 15 is not equivalent across all cycles. A possible FUTURE policy is a common 15+ top-code (or categories) plus unchanged structural tagged-zero/status pairing; requires approval. No cap/recode applied here; 777/999 are nonresponses, not quantities.",
    "SLD010H": "MISSING in 2015 XPT and official variable index. SLQ_I publishes SLD012 calculated from SLQ300/SLQ310 sleep/wake times, rounded to half-hours; range 2-14.5, versus directly reported integer SLD010H with 12+ top-code and 77/99. Different measurement/precision/top-coding, not a rename. Candidate noted only; no substitution.",
    "RIDAGEYR": "Same age at screening in years, top-coded at 80. Scope for compatibility is the already specified age>=20; no age filter/cohort is applied in this audit.",
    "RIAGENDR": "Same released 1=male, 2=female coding and eligibility; retain source label without inferring additional gender categories.",
    "RIDRETH1": "Same five race/Hispanic-origin groups. Code 5 includes other race/multiracial and does not separately expose non-Hispanic Asian. Do not replace with RIDRETH3, whose category scheme differs.",
    "DMDEDUC2": "Same five education categories and 7/9, restricted to age>=20. Do not substitute youth DMDEDUC3.",
    "INDFMPIR": "Same dimensionless family-income/poverty-guideline ratio and >=5 top-code. Annual poverty guidelines are year-specific; this is not nominal dollar income and no additional inflation transformation is implied.",
}


def unit_for(variable):
    units = {"BMXBMI": "kg/m^2", "BMXWAIST": "cm", "BPXPLS": "beats/min",
             "LBXGH": "% HbA1c", "LBXTC": "mg/dL", "LBDHDD": "mg/dL",
             "LBXSCR": "mg/dL", "LBXSAL": "g/dL", "PAD680": "minutes/day",
             "ALQ130": "drinks/drinking day; cycle-specific top-code",
             "ALQ120Q": "frequency quantity; week/month/year unit in ALQ120U",
             "SLD010H": "hours/night on weekdays or workdays",
             "RIDAGEYR": "years; 80+ top-code", "INDFMPIR": "income/poverty ratio; 5+ top-code"}
    if variable.startswith(("BPXSY", "BPXDI")):
        return "mmHg"
    if variable.startswith("DPQ"):
        return "ordinal item points 0-3; 7/9 nonresponse"
    return units.get(variable, "categorical code; see coding_json")


def lab_details(variable, cycle):
    if variable not in {"LBXGH", "LBXTC", "LBDHDD", "LBXSCR", "LBXSAL"}:
        return "", "", ""
    notes = FEATURE_NOTES[variable]
    if variable == "LBXGH":
        method = "Ion-exchange HPLC: Tosoh G7 and G8" if cycle == CYCLES[0] else "Ion-exchange HPLC: Tosoh G8"
        limit = ("Numerical LLOD not specified; G7 reportable range 3.0-19.0%, G8 3.1-19.0%; these are not LLOD values."
                 if cycle == CYCLES[0] else "HTML states LLOD not applicable; do not confuse reportable range with LLOD.")
    elif variable in {"LBXSCR", "LBXSAL"}:
        method = "IDMS-standardized Jaffe rate" if variable == "LBXSCR" else "Bromcresol-purple endpoint"
        method += "; DxC800" if cycle != CYCLES[2] else "; DxC800 and DxC660i"
        value = "0.1 mg/dL" if variable == "LBXSCR" else "1 g/dL"
        limit = "SOP sensitivity/analytical lower bound=" + value
        limit += "; HTML also reports LLOD=" + value if cycle != CYCLES[0] else "; numerical HTML LLOD not stated"
    else:
        method = ("Enzymatic total cholesterol" if variable == "LBXTC" else "Third-generation direct HDL assay")
        method += "; Roche Modular P" if cycle == CYCLES[0] else "; Roche Modular P/Cobas 6000 transition in 2013, no further 2015 change"
        limit = ("NOT_EXPLICITLY_REPORTED in reviewed 2011 HTML and SOP; clinical reportable range is not LLOD"
                 if cycle == CYCLES[0] else ("4 mg/dL" if variable == "LBXTC" else "3 mg/dL; HTML uses label LBXHDD"))
    return method, limit, notes


def codebook(path, url, wanted):
    """Extract actual variable definitions, instructions, eligibility and code tables.

    Frequency counts and observed range endpoints are NOT coding definitions.
    Original tables are retained, with a separate normalized coding signature.
    """
    root = Document(path.read_text(encoding="utf-8-sig")).root
    result = {"url": url, "sha256": sha256(path), "variables": {}, "sections": {}, "links": []}
    for node in root.walk("div"):
        children = [c for c in node.children if isinstance(c, Node)]
        headings = [c for c in children if c.tag == "h3" and c.attrs.get("class") == "vartitle"]
        if not headings:
            continue
        heading = headings[0]
        name = heading.attrs["id"].upper()
        if name not in wanted:
            continue
        fields = {}
        for dl in node.walk("dl"):
            key = None
            for item in dl.children:
                if not isinstance(item, Node):
                    continue
                if item.tag == "dt":
                    key = item.text().rstrip(": ")
                elif item.tag == "dd" and key:
                    fields[key] = item.text()
        rows = [r for r in table_rows(node) if r and r[0] != "Code or Value"]
        signature = []
        for row in rows:
            if len(row) < 2:
                continue
            code = "OBSERVED_NUMERIC_RANGE" if row[1] == "Range of Values" else row[0]
            signature.append([code, row[1], row[4].upper() if len(row) > 4 else ""])
        result["variables"][name] = {"anchor": heading.attrs["id"], "fields": fields,
                                     "coding_signature": signature, "frequency_table": rows}
    # Documentation sections are adjacent siblings of named h2 headings.
    for parent in root.walk():
        current = None
        for child in parent.children:
            if not isinstance(child, Node):
                continue
            if child.tag == "h2":
                current = child.attrs.get("id")
                if current:
                    result["sections"].setdefault(current, [])
            elif current:
                result["sections"][current].append(child.text())
    for link in root.walk("a"):
        href = link.attrs.get("href", "")
        if href and not href.startswith("#"):
            result["links"].append({"href": href, "label": link.text()})
    return result


def extract_evidence(catalog):
    wanted = set(catalog["required_raw_variables"]) | {"SLD012", "SLQ300", "SLQ310", "ALQ120U"}
    evidence = {}
    for cycle in CYCLES:
        evidence[cycle] = {}
        for module in catalog["cycles"][cycle]["modules"]:
            path = RAW / cycle / "docs" / (module["module"] + ".htm")
            evidence[cycle][module["module"]] = codebook(path, module["documentation_url"], wanted)
    save_json(EVIDENCE, evidence)
    return evidence


def extract_pdf_evidence():
    """Optional pypdf extraction using the bundled document-analysis environment."""
    from pypdf import PdfReader

    evidence = []
    manifest = pd.read_csv(ARTIFACTS / "nhanes_official_lab_method_manifest.csv")
    terms = r"(?i)(lower limit|limit of detection|limits of detection|reportable range|analytical range|method changed|sensitivity for)"
    for record in manifest.to_dict("records"):
        path = ROOT / record["local_path"]
        reader = PdfReader(path)
        snippets = []
        for page_number, page in enumerate(reader.pages, 1):
            text = " ".join((page.extract_text() or "").split())
            for match in re.finditer(terms, text):
                snippets.append({"page": page_number,
                                 "text": text[max(0, match.start() - 60):match.end() + 650]})
        if not snippets:
            raise ValueError(f"No searchable method/limit text in PDF: {path}")
        evidence.append({"cycle": record["cycle"], "filename": path.name,
                         "sha256": sha256(path), "url": record["url"],
                         "pages": len(reader.pages), "snippets": snippets})
        print(f"PDF verified {record['cycle']}/{path.name}: {len(reader.pages)} pages", flush=True)
    save_json(ARTIFACTS / "nhanes_official_pdf_evidence.json", evidence)


def raw_frames(catalog):
    """Read individual modules only. Do not combine participants or cycles."""
    frames = {}
    for cycle in CYCLES:
        frames[cycle] = {}
        for module in catalog["cycles"][cycle]["modules"]:
            frame = pd.read_sas(RAW / cycle / "xpt" / module["filename"], format="xport")
            if "SEQN" not in frame or frame.SEQN.isna().any() or frame.SEQN.duplicated().any():
                raise ValueError(f"Invalid identifiers in {cycle}/{module['module']}")
            frames[cycle][module["module"]] = frame
        for name in [f"DPQ{i:03d}" for i in range(10, 100, 10)]:
            module = catalog["cycles"][cycle]["variable_mapping"].get(name, {}).get("module")
            if not module or name not in frames[cycle][module]:
                raise ValueError(f"STOP compatibility for {cycle}: PHQ item {name} missing")
    return frames


def all_targets(dependencies):
    targets = [dict(f, role="F4") for f in dependencies["features"]]
    targets += [{"feature": name, "source_variables": [name], "derived": False,
                 "source_file": "questionnaire.csv", "group": "target", "role": "PHQ9"}
                for name in dependencies["phq9_items"]]
    return targets


def reader_zero_audit(catalog, frames, evidence):
    """Check XPORT zero bytes independently; never replace decoded or raw values.

    The installed pandas XPORT converter maps all-zero IBM float words to 2^-260
    instead of IEEE zero. Prove each case from its original observation bytes.
    This diagnostic is separate from harmonization and preserves exact-read parity.
    """
    rows, masks = [], {}
    for cycle in CYCLES:
        for module in catalog["cycles"][cycle]["modules"]:
            name = module["module"]
            path = RAW / cycle / "xpt" / module["filename"]
            with pd.read_sas(path, format="xport", iterator=True) as reader:
                fields = reader.fields
                start, width, nobs = reader.record_start, reader.record_length, reader.nobs
            raw = np.frombuffer(path.read_bytes(), dtype=np.uint8,
                                count=nobs * width, offset=start).reshape(nobs, width)
            frame = frames[cycle][name]
            assert nobs == len(frame)
            for field in fields:
                variable = field["name"].decode()
                if variable not in {"SEQN", *catalog["required_raw_variables"]}:
                    continue
                if field["ntype"] != "numeric":
                    raise ValueError(f"Unexpected nonnumeric raw dependency: {variable}")
                position, length = field["npos"], field["field_length"]
                zero = (raw[:, position:position + length] == 0).all(axis=1)
                masks[(cycle, name, variable)] = pd.Series(zero, index=frame.SEQN.to_numpy())
                values = frame[variable]
                misread = zero & values.ne(0).to_numpy()
                assert np.all(values.to_numpy()[misread] == 2.0 ** -260), "Unrecognized IBM-zero decoding issue"
                book = evidence[cycle][name]["variables"].get(variable, {})
                zero_rows = [r for r in book.get("frequency_table", []) if r[0] == "0"]
                expected = int(zero_rows[0][2]) if zero_rows else ""
                if expected != "":
                    assert int(zero.sum()) == expected, f"Codebook/raw-zero count discrepancy: {cycle}/{variable}"
                rows.append({"cycle": cycle, "module": name, "variable": variable,
                             "pandas_version": pd.__version__, "raw_zero_byte_count": int(zero.sum()),
                             "zero_decoded_as_nonzero_count": int(misread.sum()),
                             "decoded_nonzero_zero_value": 2.0 ** -260 if misread.any() else "",
                             "codebook_zero_count_if_explicit": expected,
                             "raw_values_modified": False,
                             "note": "Independent check of all-zero IBM numeric bytes using XPORT record offsets; no rounding, conversion fix or data export performed."})
    save_csv(TABLES / "nhanes_official_reader_audit.csv", rows)
    return masks


def variable_audit(catalog, dependencies, evidence, frames):
    rows = []
    for cycle in CYCLES:
        mapping = catalog["cycles"][cycle]["variable_mapping"]
        for target in all_targets(dependencies):
            feature = target["feature"]
            note = FEATURE_NOTES.get(feature, "Same PHQ-9 item meaning, past-2-week timeframe, public age>=18, no proxies; codes 0-3, 7/9 nonresponses. No score or target is calculated.")
            for variable in target["source_variables"]:
                module = mapping.get(variable, {}).get("module")
                if not module and variable == "SLD010H":
                    module = next(r["module"] for r in catalog["cycles"][cycle]["modules"]
                                  if r["module_name"] == "Sleep Disorders")
                book = evidence[cycle].get(module, {})
                meta = book.get("variables", {}).get(variable, {})
                available = module in frames[cycle] and variable in frames[cycle][module]
                if available and not meta:
                    raise ValueError(f"Variable lacks verified codebook metadata: {cycle}/{variable}")
                values = frames[cycle][module][variable] if available else None
                fields = meta.get("fields", {})
                method, llod, lab_note = lab_details(variable, cycle)
                eligibility = fields.get("Target", "Variable absent; see notes")
                if variable == "SMQ040":
                    eligibility += "; conditional on SMQ020=1"
                if variable in {"ALQ110", "ALQ120Q", "ALQ130"}:
                    eligibility += "; conditional ALQ routing (see coding_json and module codebook)"
                rows.append({"cycle": cycle, "target_feature": feature, "raw_variable": variable,
                             "module": module or "NOT_FOUND", "available": bool(available),
                             "dtype": str(values.dtype) if available else "",
                             "n_nonmissing": int(values.notna().sum()) if available else "",
                             "unique_count": int(values.nunique()) if available else "",
                             "unit": unit_for(variable), "description": fields.get("English Text", "ABSENT"),
                             "eligibility": eligibility,
                             "codebook_url": book.get("url", "") + ("#" + meta["anchor"] if meta else ""),
                             "notes": note, "derived": target["derived"],
                             "source_variables": json.dumps(target["source_variables"]),
                             "feature_group": target["group"], "feature_role": target["role"],
                             "n_rows_module": len(frames[cycle][module]) if module in frames[cycle] else "",
                             "n_missing": int(values.isna().sum()) if available else "",
                             "coding_json": json.dumps(meta.get("coding_signature", []), ensure_ascii=False),
                             "english_instructions": fields.get("English Instructions", ""),
                             "module_eligible_sample": " ".join(book.get("sections", {}).get("Eligible_Sample", [])),
                             "analytic_method": method, "lower_detection_limit": llod,
                             "codebook_sha256": book.get("sha256", "")})
    assert len({r["target_feature"] for r in rows if r["feature_role"] == "F4"}) == 34
    save_csv(TABLES / "multicycle_variable_audit.csv", rows)
    return rows


def compatibility_matrix(dependencies, audit):
    """Apply the explicit semantic review to verified raw availability.

    DIFFERENT_DEFINITION conservatively includes non-equivalent measurement
    wording/probes. It is a review flag, not a claim of irreparable incompatibility.
    """
    assert set(FEATURE_NOTES) == {x["feature"] for x in dependencies["features"]}
    rows = []
    for target in all_targets(dependencies):
        feature = target["feature"]
        default = "DERIVED_COMPATIBLE" if target["derived"] else "EXACT"
        statuses = {c: default for c in CYCLES}
        for cycle in CYCLES:
            items = [r for r in audit if r["target_feature"] == feature and r["cycle"] == cycle]
            if not all(r["available"] for r in items):
                statuses[cycle] = "MISSING"
        if feature == "alcohol_quantity_code":
            statuses = dict.fromkeys(CYCLES, "COMPATIBLE_WITH_RECODE")
        if feature == "PAQ605":
            statuses[CYCLES[0]] = "DIFFERENT_DEFINITION"
        if feature == "PAD680":
            statuses[CYCLES[0]] = statuses[CYCLES[2]] = "DIFFERENT_DEFINITION"
        priority = ["MISSING", "DIFFERENT_DEFINITION", "COMPATIBLE_WITH_RECODE", "DERIVED_COMPATIBLE", "EXACT"]
        overall = next(s for s in priority if s in statuses.values())
        same_eligibility = feature not in {"smoking_status", "BPXPLS"}
        # Age Targets match for ALQ, but module-level exclusion wording is incomplete in 2011.
        if feature in {"ALQ101", "alcohol_status", "alcohol_quantity_code"}:
            same_eligibility = "NOT_FULLY_DOCUMENTED"
        absent = overall == "MISSING"
        rows.append({"feature": feature, **statuses,
                     "same_definition": False if feature in {"PAQ605", "PAD680"} else ("NOT_COMPARABLE" if absent else True),
                     "same_unit": "NOT_COMPARABLE" if absent else True,
                     "same_coding": "NOT_COMPARABLE" if absent else feature != "alcohol_quantity_code",
                     "same_eligibility": "NOT_COMPARABLE" if absent else same_eligibility,
                     "harmonization_needed": overall in {"MISSING", "DIFFERENT_DEFINITION", "COMPATIBLE_WITH_RECODE"},
                     "harmonization_note": FEATURE_NOTES.get(feature, "Same 0-3 PHQ-9 coding, 7/9 nonresponse, two-week timeframe and adult eligibility. Minor brackets/hand-card spelling do not change item meaning."),
                     "feature_role": target["role"], "overall_status": overall,
                     "comparison_scope": "Existing age>=20 framework; semantic definitions, not identical documentation wording",
                     "same_age_eligibility_within_age20": "NOT_COMPARABLE" if absent else True,
                     "source_variables": json.dumps(target["source_variables"])})
    save_csv(TABLES / "multicycle_feature_compatibility.csv", rows)
    return rows


def compare_2013_parity(catalog, dependencies, frames, zero_masks):
    """Compare each raw variable by common SEQN, never construct a merged cohort."""
    variable_source = {v: f["source_file"] for f in dependencies["features"] for v in f["source_variables"]}
    variable_source.update({v: "questionnaire.csv" for v in dependencies["phq9_items"]})
    old = {}
    for filename in sorted(set(variable_source.values())):
        columns = ["SEQN"] + sorted(v for v, source in variable_source.items() if source == filename)
        frame = pd.read_csv(ROOT / "data/raw" / filename, usecols=columns, float_precision="round_trip")
        if frame.SEQN.isna().any() or frame.SEQN.duplicated().any():
            raise ValueError(f"Invalid SEQN in old raw source {filename}")
        old[filename] = frame.set_index("SEQN", drop=False)
    cycle = "2013_2014"
    rows, mismatch_details, id_seen = [], [], set()
    for variable in sorted(variable_source):
        module = catalog["cycles"][cycle]["variable_mapping"][variable]["module"]
        source = variable_source[variable]
        kaggle = old[source]
        official = frames[cycle][module].set_index("SEQN", drop=False)
        common = kaggle.index.intersection(official.index).sort_values()
        if not len(common):
            raise ValueError(f"No common SEQN for {variable}")
        compare_names = [variable]
        if (module, source) not in id_seen:
            compare_names.insert(0, "SEQN")
            id_seen.add((module, source))
        for name in compare_names:
            left, right = kaggle.loc[common, name], official.loc[common, name]
            both_missing = left.isna() & right.isna()
            both_present = left.notna() & right.notna()
            exact = left.eq(right) | both_missing
            verified_zero = zero_masks[(cycle, module, name)].loc[common]
            zero_artifact = ~exact & left.eq(0) & verified_zero
            source_exact = exact | zero_artifact
            missing_match = left.isna().eq(right.isna())
            difference = (left[both_present] - right[both_present]).abs()
            numeric_tolerance_matches = int(difference.le(1e-10).sum() + both_missing.sum())
            rows.append({"variable": name, "module": module, "kaggle_source": source,
                         "n_common_SEQN": len(common), "exact_match_count": int(exact.sum()),
                         "exact_match_percent": float(100 * exact.mean()),
                         "missing_pattern_match": bool(missing_match.all()),
                         "mismatch_count": int((~exact).sum()),
                         "max_absolute_difference": float(difference.max()) if len(difference) else "",
                         "missing_pattern_match_count": int(missing_match.sum()),
                         "missing_pattern_match_percent": float(100 * missing_match.mean()),
                         "both_missing_count": int(both_missing.sum()),
                         "both_nonmissing_count": int(both_present.sum()),
                         "nonmissing_exact_match_count": int((exact & both_present).sum()),
                         "match_count_at_abs_tolerance_1e_minus10": numeric_tolerance_matches,
                         "mismatch_count_at_abs_tolerance_1e_minus10": len(common) - numeric_tolerance_matches,
                         "verified_raw_zero_decoder_mismatch_count": int(zero_artifact.sum()),
                         "source_value_exact_match_count": int(source_exact.sum()),
                         "source_value_exact_match_percent": float(100 * source_exact.mean()),
                         "source_value_mismatch_count": int((~source_exact).sum()),
                         "n_kaggle_SEQN": len(kaggle), "n_official_SEQN": len(official),
                         "n_kaggle_only_SEQN": len(kaggle.index.difference(official.index)),
                         "n_official_only_SEQN": len(official.index.difference(kaggle.index)),
                         "comparison_rule": "exact_match uses unchanged read_sas output; both missing match. source_value_exact also accepts Kaggle 0 only where original XPT numeric bytes are all zero. Absolute-tolerance diagnostic is separate; no values modified."})
            for seqn in common[~source_exact]:
                mismatch_details.append({"SEQN": int(seqn), "variable": name, "module": module,
                                         "kaggle_value": float(left.loc[seqn]),
                                         "official_read_sas_value": float(right.loc[seqn]),
                                         "note": "Actual source-value mismatch, not an IBM-zero reader artifact. Both datasets remain unchanged."})
    save_csv(TABLES / "kaggle_vs_official_2013_parity.csv", rows)
    if mismatch_details:
        save_csv(TABLES / "kaggle_vs_official_2013_mismatch_details.csv", mismatch_details)
    return rows


def write_lab_audit(evidence):
    rows = []
    method_manifest = pd.read_csv(ARTIFACTS / "nhanes_official_lab_method_manifest.csv")
    for cycle in CYCLES:
        for variable in ("LBXGH", "LBXTC", "LBDHDD", "LBXSCR", "LBXSAL"):
            module, book = next((m, b) for m, b in evidence[cycle].items() if variable in b["variables"])
            method, llod, notes = lab_details(variable, cycle)
            labels = {"LBXGH": "Glycohemoglobin", "LBXTC": "Total Cholesterol",
                      "LBDHDD": "HDL", "LBXSCR": "Creatinine", "LBXSAL": "Albumin"}
            methods = method_manifest[method_manifest.cycle.eq(cycle) & method_manifest.analyte.str.startswith(labels[variable])]
            rows.append({"cycle": cycle, "variable": variable, "module": module,
                         "unit": unit_for(variable), "analytic_method": method,
                         "lower_detection_limit": llod, "notes": notes,
                         "codebook_url": book["url"],
                         "method_urls": json.dumps(methods.url.tolist()),
                         "method_local_paths": json.dumps(methods.local_path.tolist())})
    save_csv(TABLES / "nhanes_official_lab_audit.csv", rows)


def summarize(dependencies, audit, compatibility, parity, downloader):
    verified = downloader.verify_all()
    preserved = check_preservation()
    f4 = [r for r in compatibility if r["feature_role"] == "F4"]
    variables = [r for r in parity if r["variable"] != "SEQN"]
    ledger = list(downloader.ledger.values())
    summary = {"stage": "DOWNLOAD_AND_DATA_AUDIT_ONLY", "finished_at": utc_now(),
               "f4_features": len(f4), "unique_raw_f4_dependencies": len({v for f in dependencies["features"] for v in f["source_variables"]}),
               "compatibility_counts": dict(pd.Series([r["overall_status"] for r in f4]).value_counts().items()),
               "all_cycles_have_nine_phq_items": all(r["available"] for r in audit if r["feature_role"] == "PHQ9"),
               "xpt_files": sum(r["kind"] == "xpt" for r in ledger),
               "xpt_size_bytes": sum(r["size_bytes"] for r in ledger if r["kind"] == "xpt"),
               "documentation_size_bytes": sum(r["size_bytes"] for r in ledger if r["kind"] != "xpt"),
               "total_download_bytes": sum(r["size_bytes"] for r in ledger),
               "downloaded_files_hash_verified": verified, "preexisting_files_unchanged": preserved,
               "parity_variables": len(variables), "parity_variables_100_percent_exact": sum(r["mismatch_count"] == 0 for r in variables),
               "parity_missing_pattern_all_match": all(r["missing_pattern_match"] for r in variables),
               "parity_exact_mismatches": sum(r["mismatch_count"] for r in variables),
               "parity_verified_zero_reader_artifacts": sum(r["verified_raw_zero_decoder_mismatch_count"] for r in variables),
               "parity_source_value_mismatches": sum(r["source_value_mismatch_count"] for r in variables),
               "parity_variables_source_value_exact": sum(r["source_value_mismatch_count"] == 0 for r in variables),
               "parity_mismatches_above_1e_minus10": sum(r["mismatch_count_at_abs_tolerance_1e_minus10"] for r in variables),
               "ready_for_exact_unchanged_34_feature_multicycle": False,
               "blocked_on": ["SLD010H missing in 2015-2016; no SLD012 substitution authorized",
                              "ALQ130/alcohol_quantity_code top-code policy needs approval",
                              "PAQ605 wording scope and PAD680 probing require a documented comparability decision",
                              "Validate/fix the XPORT zero reader before future PHQ scoring or zero-dependent routing"],
               "additional_cautions": ["2011 HDL/total-cholesterol numerical LLODs not explicitly documented in reviewed sources",
                                       "pandas read_sas maps all-zero IBM words to 2^-260 in this environment; verified bytewise. Fix/validate the reader before future scoring or zero-dependent routing, not by silently normalizing during this audit",
                                       "One real PAQ665 source mismatch in 2013 (SEQN 73756); old experiments/dataset remain unchanged",
                                       "Lab instrument/site transitions; do not infer assay interchangeability from identical units alone",
                                       "NHANES survey weights and cycle effects need a future analysis protocol; not implemented"],
               "prohibited_actions_performed": [],
               "statistics_scope": "Each entire raw module; includes nonresponse codes as nonmissing. No age filtering or PHQ/cohort selection.",
               "compatibility_semantics": "EXACT is semantic variable compatibility within age>=20, not identical measurement error/instruments/wording. DIFFERENT_DEFINITION is a conservative wording/probe review flag. Derived compatibility is assessed from code, without computing derived values.",
               "codebook_evidence_sha256": sha256(EVIDENCE),
               "pdf_evidence_sha256": sha256(ARTIFACTS / "nhanes_official_pdf_evidence.json")}
    summary["compatibility_counts"] = {k: int(v) for k, v in summary["compatibility_counts"].items()}
    outputs = ["multicycle_variable_audit.csv", "multicycle_feature_compatibility.csv",
               "kaggle_vs_official_2013_parity.csv", "nhanes_official_lab_audit.csv",
               "nhanes_official_xpt_verification.csv", "nhanes_official_reader_audit.csv",
               "kaggle_vs_official_2013_mismatch_details.csv"]
    summary["output_sha256"] = {f"results/tables/{p}": sha256(TABLES / p) for p in outputs}
    save_json(ARTIFACTS / "nhanes_official_audit_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--pdf-evidence-only", action="store_true")
    args = parser.parse_args()
    check_preservation()
    downloader = Downloader()
    downloader.verify_all()
    if args.pdf_evidence_only:
        extract_pdf_evidence()
        return
    catalog = json.loads((ARTIFACTS / "nhanes_official_source_catalog.json").read_text())
    evidence = extract_evidence(catalog)
    print(f"Codebooks extracted for {sum(len(c) for c in evidence.values())} modules")
    if args.extract_only:
        return
    reviewed_digest = "098bf974567996f4b1d02800006b0fe4e2c45e0dfe031fa418f7b2a54b3fc17d"
    if sha256(EVIDENCE) != reviewed_digest:
        raise ValueError("Codebooks differ from the semantic review; review before assigning compatibility")
    dependencies = json.loads((ARTIFACTS / "nhanes_official_feature_dependencies.json").read_text())
    for name, digest in dependencies["source_sha256"].items():
        assert sha256(ROOT / name) == digest, f"Frozen dependency source changed: {name}"
    frames = raw_frames(catalog)
    zero_masks = reader_zero_audit(catalog, frames, evidence)
    audit = variable_audit(catalog, dependencies, evidence, frames)
    compatibility = compatibility_matrix(dependencies, audit)
    parity = compare_2013_parity(catalog, dependencies, frames, zero_masks)
    write_lab_audit(evidence)
    summarize(dependencies, audit, compatibility, parity, downloader)


if __name__ == "__main__":
    main()
