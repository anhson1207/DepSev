"""Summarize saved A/B/C/D predictions only; never train or select a model."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import multicycle_protocol as p


def saved_records():
    records = []
    for experiment, seeds in [("A", [0]), ("B", [0]), ("C", p.SEEDS), ("D", [0])]:
        for seed in seeds:
            for task in p.TASKS:
                path = p.ART / "checkpoints" / f"{experiment}_{seed}_{task}.json"
                record = p.read_json(path)
                assert record["config_sha256"] == p.digest(p.ART / "experiment_config.json")
                assert record["predictions_sha256"] == p.digest(path.with_name(f"{path.stem}_predictions.csv"))
                records.append(record)
    return records


def comparison_table(summaries):
    rows = []
    for task in p.TASKS:
        part = summaries.loc[summaries.target.eq(task)]
        for metric in p.METRICS:
            if task == "T2" and metric in p.METRICS[-2:]:
                continue
            row = {"target": task, "metric": metric}
            for experiment, label in [("A", "A_single"), ("C", "C_matched"), ("B", "B_full"), ("D", "D_temporal")]:
                values = part.loc[part.experiment.eq(experiment)]
                row[label] = values[f"{metric}_mean"].mean()
                row[f"{label}_sd"] = values[f"{metric}_mean"].std(ddof=1) if experiment == "C" else values[f"{metric}_std"].iloc[0]
            row["delta_B_minus_A"] = row["B_full"] - row["A_single"]
            row["delta_C_minus_A"] = row["C_matched"] - row["A_single"]
            rows.append(row)
    return pd.DataFrame(rows)


def rare_comparison(classes):
    rows = []
    for label in [3, 4]:
        for experiment in ["A", "B", "C", "D"]:
            values = classes.loc[classes.target.eq("T5") & classes.class_index.eq(label) & classes.experiment.eq(experiment)]
            row = {"experiment": experiment, "class": f"C{label}",
                   "aggregation": "mean over 10 sample-level OOF results" if experiment == "C" else "temporal test" if experiment == "D" else "pooled OOF"}
            for metric in ["precision", "recall", "f1", "support", "predicted_count", "true_positives"]:
                row[metric] = values[metric].mean()
                row[f"{metric}_sd"] = values[metric].std(ddof=1) if experiment == "C" else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def pooled_cycle_metrics():
    predictions = pd.read_csv(p.TABLES / "B_oof_predictions.csv")
    rows, per_class = [], []
    for (cycle, task), group in predictions.groupby(["cycle", "target"]):
        metrics, classes, _ = p.score(group.y_true, group.y_pred, task)
        row = {"cycle": cycle, "target": task, "n": len(group), "scope": "pooled-trained OOF subset", **metrics}
        row.update({f"recall_class_{c['class_index']}": c["recall"] for c in classes})
        rows.append(row)
        per_class.extend({"cycle": cycle, "target": task, **c} for c in classes)
    return pd.DataFrame(rows), pd.DataFrame(per_class)


def pyplot():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "depsev-multicycle-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def plots(comparison, rare, confusion):
    plt = pyplot()
    p.FIGURES.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig, name):
        path = p.FIGURES / name
        with path.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=170, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    for metric, label in [("macro_f1", "Macro-F1"), ("balanced_accuracy", "Balanced accuracy")]:
        part = comparison.loc[comparison.metric.eq(metric)].set_index("target").loc[list(p.TASKS)]
        fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
        for prefix, name, color in [("A_single", "A: official 2013-2014 (N=5,054)", "#245b8f"),
                                    ("B_full", "B: pooled 2011-2016 (N=14,581)", "#be5a22")]:
            ax.errorbar(part.index, part[prefix], yerr=part[f"{prefix}_sd"], marker="o", capsize=4, color=color, label=name)
        ax.set(ylim=(0, 1), ylabel=label, xlabel="Target granularity", title=f"Frozen RF/common31: {label}\nMean ± sample SD across five folds")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=.2)
        save(fig, f"{metric}_A_vs_B.png")
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for i, label in enumerate(["C3", "C4"]):
        part = rare.loc[rare["class"].eq(label)].set_index("experiment").loc[list("ABCD")]
        ax.bar(np.arange(4) + (i - .5) * .32, part.recall, width=.32, label=label, color=["#245b8f", "#be5a22"][i])
        # Only C has across-sample SD; absence of other error bars is deliberate.
        ax.errorbar(2 + (i - .5) * .32, part.loc["C", "recall"], yerr=part.loc["C", "recall_sd"], fmt="none", color="black", capsize=4)
    ax.set(xticks=range(4), xticklabels=["A: single", "B: pooled", "C: matched-N", "D: temporal"],
           ylim=(0, 1), ylabel="Recall (T5)", title="Rare-class recall: OOF for A/B/C, temporal test for D\nC bars: mean ± sample SD across 10 samples")
    ax.legend()
    save(fig, "rare_class_recall.png")
    for experiment in ["A", "B", "D"]:
        for task in p.TASKS:
            part = confusion.loc[confusion.experiment.eq(experiment) & confusion.target.eq(task)]
            matrix = part.pivot(index="true_class", columns="predicted_class", values="count").to_numpy()
            proportions = matrix / matrix.sum(axis=1, keepdims=True)
            k = len(matrix)
            fig, ax = plt.subplots(figsize=(6.4, 5.3), layout="constrained")
            im = ax.imshow(proportions, cmap="Blues", vmin=0, vmax=1)
            for i in range(k):
                for j in range(k):
                    ax.text(j, i, f"{matrix[i,j]}\n{proportions[i,j]:.1%}", ha="center", va="center", color="white" if proportions[i,j] > .55 else "#202020")
            labels = [f"C{i}" if task == "T5" else f"{task}_{i}" for i in range(k)]
            ax.set(xticks=range(k), yticks=range(k), xticklabels=labels, yticklabels=labels,
                   xlabel="Predicted class", ylabel="True class", title=f"{experiment} / {task}: {'temporal test' if experiment == 'D' else 'OOF'} (N={matrix.sum():,})\nCounts and row percentages")
            fig.colorbar(im, ax=ax, label="Fraction within true class", shrink=.85)
            save(fig, f"{experiment}_{task}_confusion.png")
    return paths


def markdown_table(frame):
    """Small plain Markdown table without adding a tabulate dependency."""
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "—" if pd.isna(value) else f"{value:.4f}"
        return str(value).replace("|", "/")
    return "\n".join(["| " + " | ".join(frame.columns) + " |",
                      "| " + " | ".join(["---"] * len(frame.columns)) + " |",
                      *["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]])


def technical_summary(comparison, rare, cycles, deltas, patterns):
    lines = ["# Controlled multi-cycle experiment: RF/F4_common31", "",
             "## Protocol and interpretation", "",
             "Official CDC cohorts only; age >=20 and nine valid PHQ-9 items. A: 2013–2014, N=5,054. B: pooled 2011–2016, N=14,581. C: 10 exact-class-count matched samples, each N=5,054, from 2011–2012 and 2015–2016 only. D: train 2011–2014, N=9,693; temporal test 2015–2016, N=4,888.", "",
             "Frozen 31 predictors and RF configuration. Numerical median imputation + StandardScaler; categorical constant missing_or_unknown + one-hot encoding. Fresh preprocessing inside every training partition. T5-stratified five-fold assignments (shuffle=True, seed=42) reused for T2–T5 within each dataset/sample. No cycle, PHQ items, total, severity or applicability metadata in X. No tuning, resampling of training folds, or threshold adjustment.", "",
             "T2: 0–4 / 5–27 (minimal symptoms vs symptomatic); T3: 0–4 / 5–9 / 10–27; T4: 0–4 / 5–9 / 10–14 / 15–27; T5: 0–4 / 5–9 / 10–14 / 15–19 / 20–27.", "",
             "A/B means and SD describe five validation folds. C means and SD describe 10 sample-level CV means, not 50 independent folds. D is one temporal test, without CV SD. Per-class A/B results use pooled OOF predictions; C averages 10 sample-level OOF metrics. No confidence intervals or paired A-vs-B significance tests are used.", "",
             "## Main comparisons", ""]
    for metric in ["macro_f1", "balanced_accuracy"]:
        part = comparison.loc[comparison.metric.eq(metric)]
        display = pd.DataFrame({"Target": part.target})
        for label in ["A_single", "C_matched", "B_full"]:
            display[label] = [f"{v:.4f} ± {sd:.4f}" for v, sd in zip(part[label], part[f"{label}_sd"])]
        display["D_temporal"] = part.D_temporal
        display["B_minus_A"] = part.delta_B_minus_A
        lines += [f"### {metric}", "", markdown_table(display), ""]
    lines += ["## T5 rare classes", "", markdown_table(rare[["experiment", "class", "recall", "recall_sd", "f1", "f1_sd", "support", "true_positives"]]), "",
              "## Confusion patterns", "", "C below aggregates prediction occurrences across overlapping samples, not unique participants.", "",
              markdown_table(patterns), "", "## Pooled-model OOF performance by source cycle", "",
              markdown_table(cycles[["cycle", "target", "n", "macro_f1", "balanced_accuracy"]]), "",
              "## Granularity deltas", "", markdown_table(deltas), "",
              "## Ordinal behaviour", "",
              markdown_table(comparison.loc[comparison.metric.isin(p.METRICS[-2:]), ["target", "metric", "A_single", "C_matched", "B_full", "D_temporal"]]), "",
              "Ordinal distance is measured in task-specific class indices, not PHQ points. T3/T4/T5 units differ; absolute MAE cannot be compared as an identical unit across these tasks.", "",
              "## Limitations", "",
              "- A and B differ in participants and overlap because B contains A. D test participants also occur in B/C analyses. These are separate frozen evaluations, not independent replications, and D must not be described as an untouched external dataset.",
              "- C samples overlap each other. Their SD captures sampling sensitivity conditional on this external-cycle pool, not an independent-sample confidence interval. Matching N and class counts does not match all demographics, covariates or measurement conditions.",
              "- RF hyperparameters were previously selected in the legacy 2013–2014 project. Reusing them controls the algorithm here but is not nested-CV evaluation of the entire historical selection process. Legacy 34-feature performance is not used as the control.",
              "- Survey weights, clustering and strata are not incorporated; results describe the complete-PHQ adult cohorts, not population prevalence or design-based uncertainty. Random CV is not household/PSU-grouped.",
              "- Laboratory platform/site changes, questionnaire context, missingness and distribution shifts may contribute to cycle differences. Sparse severe classes remain a limitation; overlap and imbalance effects are not isolated.",
              "- The class_weight=balanced_subsample rule is fixed, but effective bootstrap weights change with target granularity and class frequencies. No training-fold over/undersampling was used; C cohort sampling is the prespecified matched control.",
              "- Associations between larger pooled data and performance are descriptive, not causal evidence. No model/configuration is changed in response to these results.", "",
              "## Files", "", "All tables are in `tables/`, all figures in `figures/`, and protocol/checkpoint hashes in `artifacts/multicycle/`. Existing Kaggle results, model artifacts, main report and README are unchanged.", ""]
    path = p.OUT / "multicycle_experiment_summary.md"
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    return path


def main():
    completion = p.ART / "summary_complete.json"
    if completion.exists():
        p.verify_hashes(p.read_json(completion)["output_sha256"])
        print("Verified existing summaries; no overwrite.")
        return
    records = saved_records()
    summaries = pd.DataFrame([r["summary"] for r in records])
    classes = pd.DataFrame([row for record in records for row in record["per_class"]])
    confusion = pd.DataFrame([row for record in records for row in record["confusion"]])
    aggregate_c = confusion.loc[confusion.experiment.eq("C")].groupby(["target", "true_class", "predicted_class"], as_index=False)["count"].sum()
    aggregate_c["row_proportion"] = aggregate_c["count"] / aggregate_c.groupby(["target", "true_class"])["count"].transform("sum")
    aggregate_c["experiment"], aggregate_c["sampling_seed"], aggregate_c["scope"] = "C", -1, "sum_10_OOF_prediction_occurrences_not_unique_participants"
    comparison, rare = comparison_table(summaries), rare_comparison(classes)
    cycles, cycle_classes = pooled_cycle_metrics()
    delta_rows = []
    for experiment, label in [("A", "A_single"), ("B", "B_full"), ("D", "D_temporal")]:
        for metric in ["macro_f1", "balanced_accuracy"]:
            values = comparison.loc[comparison.metric.eq(metric)].set_index("target")[label]
            for before, after in zip(["T2", "T3", "T4"], ["T3", "T4", "T5"]):
                delta_rows.append({"experiment": experiment, "metric": metric, "transition": f"{after}-{before}", "delta": values[after] - values[before]})
    deltas = pd.DataFrame(delta_rows)
    patterns = []
    for experiment in ["A", "B", "C", "D"]:
        part = (aggregate_c if experiment == "C" else confusion.loc[confusion.experiment.eq(experiment)])
        part = part.loc[part.target.eq("T5")]
        for a, b in [(1, 0), (3, 2), *[(4, j) for j in range(5)]]:
            row = part.loc[part.true_class.eq(a) & part.predicted_class.eq(b)].iloc[0]
            support = part.loc[part.true_class.eq(a), "count"].sum()
            patterns.append({"experiment": experiment, "pattern": f"C{a}->C{b}", "count": int(row["count"]), "support": int(support), "percent": 100 * row.row_proportion})
    patterns = pd.DataFrame(patterns)
    tables = {"comparison_summary.csv": comparison, "granularity_deltas.csv": deltas,
              "per_class_metrics.csv": classes, "pooled_per_cycle_metrics.csv": cycles,
              "pooled_per_cycle_per_class_metrics.csv": cycle_classes,
              "confusion_matrices.csv": pd.concat([confusion, aggregate_c], ignore_index=True),
              "rare_class_comparison.csv": rare, "T5_confusion_patterns.csv": patterns}
    outputs = []
    for name, table in tables.items():
        path = p.TABLES / name
        p.write_csv(path, table)
        outputs.append(path)
    audit_path = p.ART / "preprocessing_fit_audit.json"
    p.write_json(audit_path, [row for record in records for row in record["fit_audits"]])
    outputs.append(audit_path)
    outputs.extend(plots(comparison, rare, confusion))
    outputs.append(technical_summary(comparison, rare, cycles, deltas, patterns))
    p.preserve_legacy()
    p.write_json(completion, {"config_sha256": p.digest(p.ART / "experiment_config.json"),
                             "output_sha256": {str(path.relative_to(p.ROOT)): p.digest(path) for path in outputs}})
    print(comparison.loc[comparison.metric.isin(["macro_f1", "balanced_accuracy"])].to_string(index=False))


if __name__ == "__main__":
    main()
