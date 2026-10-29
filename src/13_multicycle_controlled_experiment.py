"""A/B/C controlled RF/common31 CV; never execute or overwrite legacy results."""

import argparse

import numpy as np
import pandas as pd

import multicycle_protocol as p


def prepare_design(frames):
    single, pooled = frames["2013_2014"], frames["2011_2016_common31"]
    counts = single.severity.value_counts().sort_index().to_numpy()
    pool = pooled.loc[~pooled.cycle.eq("2013_2014")].reset_index(drop=True)
    samples = {seed: p.matched_sample(pool, counts, seed) for seed in p.SEEDS}
    designs = [("A", 0, single), ("B", 0, pooled), *[("C", seed, sample) for seed, sample in samples.items()]]
    manifest, fold_rows = [], []
    for experiment, seed, frame in designs:
        folds = p.fixed_folds(frame)
        assignment = frame[["SEQN", "cycle", "severity"]].copy()
        assignment["experiment"], assignment["sampling_seed"], assignment["fold"] = experiment, seed, folds
        fold_rows.append(assignment)
        if experiment == "C":
            manifest.append(assignment)
            assert not set(p.participant_keys(frame)) & set(p.participant_keys(single))
            np.testing.assert_array_equal(frame.severity.value_counts().sort_index(), counts)
    manifest, fold_rows = pd.concat(manifest, ignore_index=True), pd.concat(fold_rows, ignore_index=True)
    for name, table in [("matched_n_sampling_manifest.csv", manifest), ("cv_fold_assignments.csv", fold_rows)]:
        path = p.TABLES / name
        if path.exists():
            pd.testing.assert_frame_equal(pd.read_csv(path), table, check_dtype=False)
        else:
            p.write_csv(path, table)
    return designs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    frames, policy, old = p.load_inputs()
    config = p.freeze_config(frames, policy, old)
    designs = prepare_design(frames)
    print(f"SANITY PASS: A=5054 B=14581 C=10x5054; features=31; globally unique SEQN={config['global_seqn_unique']}; legacy files={p.preserve_legacy()}", flush=True)
    if args.audit_only:
        print("Audit only: no estimator fitted.", flush=True)
        return
    completion = p.ART / "cv_complete.json"
    if completion.exists():
        p.verify_hashes(p.read_json(completion)["output_sha256"])
        print("Verified completed A/B/C results; no rerun.", flush=True)
        return
    records, predictions = [], []
    for experiment, seed, frame in designs:
        for task in p.TASKS:
            record, prediction = p.run_task(experiment, frame, task, policy, old, seed)
            records.append(record)
            predictions.append(prediction)
    summaries = pd.DataFrame([record["summary"] for record in records])
    all_predictions = pd.concat(predictions, ignore_index=True)
    outputs = []
    for experiment, name in [("A", "single_cycle_cv_metrics.csv"), ("B", "full_multicycle_cv_metrics.csv"), ("C", "matched_n_metrics.csv")]:
        path = p.TABLES / name
        p.write_csv(path, summaries.loc[summaries.experiment.eq(experiment)])
        outputs.append(path)
        path = p.TABLES / f"{experiment}_oof_predictions.csv"
        p.write_csv(path, all_predictions.loc[all_predictions.experiment.eq(experiment)])
        outputs.append(path)
    path = p.TABLES / "cv_fold_metrics.csv"
    p.write_csv(path, pd.DataFrame([row for record in records for row in record["folds"]]))
    outputs.append(path)
    p.preserve_legacy()
    p.write_json(completion, {"config_sha256": p.digest(p.ART / "experiment_config.json"),
                             "n_model_fits": sum(len(record["folds"]) for record in records),
                             "output_sha256": {str(path.relative_to(p.ROOT)): p.digest(path) for path in outputs}})
    print("A/B/C complete: 240 CV fits. No temporal prediction in this runner.", flush=True)


if __name__ == "__main__":
    main()
