"""D: frozen RF train 2011-2014, evaluate 2015-2016; no test-driven selection."""

import argparse

import pandas as pd

import multicycle_protocol as p


def temporal_partition(frames):
    train = pd.concat([frames[c] for c in p.CYCLES[:2]], ignore_index=True)
    test = frames[p.CYCLES[2]]
    assert len(train) == 9693 and len(test) == 4888
    assert not set(p.participant_keys(train)) & set(p.participant_keys(test))
    assert set(train.cycle) == set(p.CYCLES[:2]) and set(test.cycle) == {p.CYCLES[2]}
    return train, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    frames, policy, old = p.load_inputs()
    p.freeze_config(frames, policy, old)
    train, test = temporal_partition(frames)
    print("Temporal sanity PASS: train=9693, test=4888, disjoint cycle+SEQN.", flush=True)
    if args.audit_only:
        return
    completion = p.ART / "temporal_complete.json"
    if completion.exists():
        p.verify_hashes(p.read_json(completion)["output_sha256"])
        print("Verified saved temporal results; no repeat prediction.", flush=True)
        return
    records, predictions = [], []
    for task in p.TASKS:
        record, pred = p.run_task("D", test, task, policy, old, train_frame=train)
        records.append(record)
        predictions.append(pred)
    outputs = [p.TABLES / "temporal_metrics.csv", p.TABLES / "D_temporal_predictions.csv"]
    p.write_csv(outputs[0], pd.DataFrame([record["summary"] for record in records]))
    p.write_csv(outputs[1], pd.concat(predictions, ignore_index=True))
    p.preserve_legacy()
    p.write_json(completion, {"config_sha256": p.digest(p.ART / "experiment_config.json"),
                             "n_model_fits": 4, "prediction_calls_per_target": 1,
                             "output_sha256": {str(path.relative_to(p.ROOT)): p.digest(path) for path in outputs}})
    print("D complete; no model changed after evaluation.", flush=True)


if __name__ == "__main__":
    main()
