"""Shared, audited configuration for the DepSev-ML pipeline."""

from pathlib import Path


RANDOM_STATE = 42
TEST_SIZE = 0.20
N_CV_SPLITS = 5
PRIMARY_MIN_AGE = 20
PROTOCOL_VERSION = "original_split_preserved_age20"
EXPERIMENT_SUFFIX = "_final_protocol"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def experiment_path(directory, filename):
    """Keep final-protocol outputs separate from all exploratory files."""
    name = Path(filename)
    return directory / f"{name.stem}{EXPERIMENT_SUFFIX}{name.suffix}"

PHQ9_COLUMNS = [
    "DPQ010",
    "DPQ020",
    "DPQ030",
    "DPQ040",
    "DPQ050",
    "DPQ060",
    "DPQ070",
    "DPQ080",
    "DPQ090",
]

SEVERITY_LABELS = {
    0: "Minimal",
    1: "Mild",
    2: "Moderate",
    3: "Moderately severe",
    4: "Severe",
}

FEATURE_SPECS = [
    {
        "feature": "BMXBMI",
        "description": "Body mass index (kg/m²)",
        "group": "P",
        "feature_type": "numerical",
    },
    {
        "feature": "BMXWAIST",
        "description": "Waist circumference (cm)",
        "group": "P",
        "feature_type": "numerical",
    },
    {
        "feature": "BPXPLS",
        "description": "60-second pulse rate",
        "group": "P",
        "feature_type": "numerical",
    },
    {
        "feature": "SBP_mean",
        "description": "Mean of available systolic blood pressure measurements",
        "group": "P",
        "feature_type": "numerical",
        "source_columns": ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"],
    },
    {
        "feature": "DBP_mean",
        "description": "Mean of available diastolic blood pressure measurements",
        "group": "P",
        "feature_type": "numerical",
        "source_columns": ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"],
    },
    {
        "feature": "LBXGH",
        "description": "Glycohemoglobin (HbA1c, %)",
        "group": "P",
        "feature_type": "numerical",
    },
    {
        "feature": "LBXTC",
        "description": "Total cholesterol (mg/dL)",
        "group": "P",
        "feature_type": "numerical",
    },
    {
        "feature": "LBDHDD",
        "description": "Direct HDL cholesterol (mg/dL)",
        "group": "P",
        "feature_type": "numerical",
    },
    {
        "feature": "LBXSCR",
        "description": "Creatinine (mg/dL)",
        "group": "P",
        "feature_type": "numerical",
    },
    {
        "feature": "LBXSAL",
        "description": "Albumin (g/dL)",
        "group": "P",
        "feature_type": "numerical",
    },
    {
        "feature": "BPQ020",
        "description": "Ever told had high blood pressure",
        "group": "C",
        "feature_type": "categorical",
    },
    {
        "feature": "DIQ010",
        "description": "Doctor told participant they have diabetes",
        "group": "C",
        "feature_type": "categorical",
    },
    {
        "feature": "MCQ160B",
        "description": "Ever told had congestive heart failure",
        "group": "C",
        "feature_type": "categorical",
    },
    {
        "feature": "MCQ160C",
        "description": "Ever told had coronary heart disease",
        "group": "C",
        "feature_type": "categorical",
    },
    {
        "feature": "MCQ160D",
        "description": "Ever told had angina",
        "group": "C",
        "feature_type": "categorical",
    },
    {
        "feature": "MCQ160E",
        "description": "Ever told had a heart attack",
        "group": "C",
        "feature_type": "categorical",
    },
    {
        "feature": "MCQ160F",
        "description": "Ever told had a stroke",
        "group": "C",
        "feature_type": "categorical",
    },
    {
        "feature": "MCQ220",
        "description": "Ever told had cancer or malignancy",
        "group": "C",
        "feature_type": "categorical",
    },
    {
        "feature": "HUQ010",
        "description": "General health condition",
        "group": "C",
        "feature_type": "categorical",
    },
    {
        "feature": "PAQ605",
        "description": "Vigorous-intensity work activity",
        "group": "L",
        "feature_type": "categorical",
    },
    {
        "feature": "PAQ620",
        "description": "Moderate-intensity work activity",
        "group": "L",
        "feature_type": "categorical",
    },
    {
        "feature": "PAQ650",
        "description": "Vigorous recreational activity",
        "group": "L",
        "feature_type": "categorical",
    },
    {
        "feature": "PAQ665",
        "description": "Moderate recreational activity",
        "group": "L",
        "feature_type": "categorical",
    },
    {
        "feature": "PAD680",
        "description": "Minutes of sedentary activity per day",
        "group": "L",
        "feature_type": "numerical",
    },
    {
        "feature": "SMQ020",
        "description": "Smoked at least 100 cigarettes in life",
        "group": "L",
        "feature_type": "categorical",
    },
    {
        "feature": "SMQ040",
        "description": "Current cigarette smoking status",
        "group": "L",
        "feature_type": "categorical",
    },
    {
        "feature": "ALQ101",
        "description": "Had at least 12 alcoholic drinks in one year",
        "group": "L",
        "feature_type": "categorical",
    },
    {
        "feature": "ALQ130",
        "description": "Average alcoholic drinks per day in past 12 months",
        "group": "L",
        "feature_type": "numerical",
    },
    {
        "feature": "SLD010H",
        "description": "Hours of sleep on weekdays or workdays",
        "group": "L",
        "feature_type": "numerical",
    },
    {
        "feature": "RIDAGEYR",
        "description": "Age in years",
        "group": "D",
        "feature_type": "numerical",
    },
    {
        "feature": "RIAGENDR",
        "description": "Sex",
        "group": "D",
        "feature_type": "categorical",
    },
    {
        "feature": "RIDRETH1",
        "description": "Race and ethnicity category",
        "group": "D",
        "feature_type": "categorical",
    },
    {
        "feature": "DMDEDUC2",
        "description": "Education level for adults",
        "group": "D",
        "feature_type": "categorical",
    },
    {
        "feature": "INDFMPIR",
        "description": "Family income-to-poverty ratio",
        "group": "D",
        "feature_type": "numerical",
    },
]

LEAKAGE_COLUMNS = set(PHQ9_COLUMNS + ["SEQN", "DPQ100", "PHQ9_total", "severity"])
