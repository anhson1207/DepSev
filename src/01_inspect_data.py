"""Audit raw NHANES 2013–2014 CSV files before cohort construction."""

from pathlib import Path

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


def main():
    """Inspect CSV files, participant IDs, and PHQ-9 item availability."""
    project_root = Path(__file__).resolve().parent.parent
    raw_data_dir = project_root / "data" / "raw"
    csv_files = sorted(
        path
        for path in raw_data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".csv"
    )

    print("NHANES 2013-2014 raw data audit")
    print(f"Raw data directory: {raw_data_dir}")
    print(f"CSV files found: {len(csv_files)}")

    if not csv_files:
        print("\nNo CSV files were found.")
        print("Extract the NHANES CSV files into data/raw/ and run this script again.")
        return

    try:
        import pandas as pd
    except ModuleNotFoundError:
        print("\npandas is not installed in the active Python environment.")
        print("Install project dependencies with: pip install -r requirements.txt")
        return

    phq9_locations = {column: [] for column in PHQ9_COLUMNS}

    for csv_file in csv_files:
        relative_path = csv_file.relative_to(project_root)
        print("\n" + "=" * 80)
        print(f"File: {relative_path}")

        try:
            data = pd.read_csv(csv_file, low_memory=False)
            encoding = "utf-8"
        except UnicodeDecodeError:
            try:
                data = pd.read_csv(csv_file, encoding="cp1252", low_memory=False)
                encoding = "cp1252"
            except Exception as error:
                print(f"Could not read file: {error}")
                continue
        except Exception as error:
            print(f"Could not read file: {error}")
            continue

        print(f"Encoding: {encoding}")
        print(f"Shape: {data.shape[0]} rows x {data.shape[1]} columns")
        print(f"Rows: {data.shape[0]}")
        print(f"Columns: {data.shape[1]}")
        print("First 20 columns:")
        print(list(data.columns[:20]))

        missing_counts = data.isna().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
        print("Columns with missing values:")
        print(missing_counts.to_string() if not missing_counts.empty else "None")

        if "SEQN" in data.columns:
            seqn = data["SEQN"]
            non_missing_seqn = seqn.dropna()
            duplicate_seqn = non_missing_seqn.duplicated(keep=False)

            print("SEQN: present")
            print(f"SEQN missing: {seqn.isna().sum()}")
            print(f"SEQN unique: {non_missing_seqn.nunique()}")
            print(
                "Duplicated SEQN values: "
                f"{non_missing_seqn[duplicate_seqn].nunique()}"
            )
            print(f"Rows with duplicate SEQN: {duplicate_seqn.sum()}")
        else:
            print("SEQN: not found")

        phq9_in_file = [column for column in PHQ9_COLUMNS if column in data.columns]
        print(f"PHQ-9 items found: {phq9_in_file or 'None'}")

        for column in phq9_in_file:
            phq9_locations[column].append(str(relative_path))
            print(f"\n{column}")
            print(f"Missing: {data[column].isna().sum()} of {len(data)}")
            print("Value counts (including missing):")
            print(data[column].value_counts(dropna=False).to_string())

        if len(phq9_in_file) == len(PHQ9_COLUMNS):
            valid_phq9_items = data[PHQ9_COLUMNS].isin([0, 1, 2, 3])
            complete_valid_phq9 = valid_phq9_items.all(axis=1)
            print("\nPHQ-9 validity summary")
            print(f"Rows in PHQ-9 file: {len(data)}")
            print(f"Rows with all 9 valid items (0-3): {complete_valid_phq9.sum()}")
            print(
                "Rows with at least one missing/invalid item: "
                f"{(~complete_valid_phq9).sum()}"
            )

    print("\n" + "=" * 80)
    print("PHQ-9 location summary")
    for column, locations in phq9_locations.items():
        location_text = ", ".join(locations) if locations else "NOT FOUND"
        print(f"{column}: {location_text}")

    missing_phq9_columns = [
        column for column, locations in phq9_locations.items() if not locations
    ]
    print(f"\nPHQ-9 items not found: {missing_phq9_columns or 'None'}")
    if missing_phq9_columns:
        raise RuntimeError(
            "Pipeline stopped because PHQ-9 items are missing: "
            + ", ".join(missing_phq9_columns)
        )


if __name__ == "__main__":
    main()
