"""Describe feature overlap among original C0-C4 on development only.

Uses observed numerical values, explicit categorical missing levels, and new
development OOF confusion. PCA preprocessing is fitted for EDA only and is never
shared with CV. Associations and marginal overlap are not causal explanations
or proofs of multivariate inseparability. Run after 09_granularity_experiment.py.
"""

from importlib import import_module

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr
from scipy.stats.contingency import association
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

import final_protocol as frozen
from config import ARTIFACTS_DIR, PROJECT_ROOT, TABLES_DIR
from protocol import sha256_file
from utils import load_json


granularity = import_module("09_granularity_experiment")
RUN_PATH = ARTIFACTS_DIR / "overlap_run.json"
PAIRS = [(i, i + 1) for i in range(4)]


def cliffs_delta(lower, higher):
    """P(lower > higher) - P(lower < higher); ties contribute zero."""
    lower, higher = np.asarray(lower), np.asarray(higher)
    if not len(lower) or not len(higher):
        return np.nan
    u = mannwhitneyu(lower, higher, alternative="two-sided", method="asymptotic").statistic
    return float(2 * u / (len(lower) * len(higher)) - 1)


def pooled_edges(values):
    """One set of development-wide decile boundaries per numerical feature."""
    edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
    if len(edges) == 1:
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    return edges


def histogram_overlap(lower, higher, edges):
    if not len(lower) or not len(higher):
        return np.nan
    a = np.histogram(lower, bins=edges)[0] / len(lower)
    b = np.histogram(higher, bins=edges)[0] / len(higher)
    return float(np.minimum(a, b).sum())


def numerical_analysis(X, y, total, names):
    summaries, correlations, pairs = [], [], []
    for name in names:
        values = X[name]
        observed = values.notna()
        rho, pvalue = spearmanr(values[observed], total[observed])
        correlations.append({"feature": name, "n_observed": int(observed.sum()),
                             "n_missing": int((~observed).sum()), "spearman_rho": rho,
                             "abs_spearman_rho": abs(rho), "p_value_exploratory_unadjusted": pvalue})
        edges = pooled_edges(values.dropna().to_numpy())
        for code in range(5):
            group = values[y.eq(code)]
            q = group.quantile([0.25, 0.5, 0.75])
            summaries.append({"feature": name, "class": f"C{code}", "n_total": len(group),
                              "n_observed": int(group.notna().sum()), "n_missing": int(group.isna().sum()),
                              "median": q.loc[0.5], "q25": q.loc[0.25], "q75": q.loc[0.75],
                              "iqr": q.loc[0.75] - q.loc[0.25]})
        for a, b in PAIRS:
            lower, higher = values[y.eq(a)].dropna(), values[y.eq(b)].dropna()
            delta = cliffs_delta(lower, higher)
            pairs.append({"class_pair": f"C{a}-C{b}", "feature": name,
                          "n_lower_observed": len(lower), "n_higher_observed": len(higher),
                          "cliffs_delta_lower_minus_higher": delta, "abs_cliffs_delta": abs(delta),
                          "histogram_overlap": histogram_overlap(lower, higher, edges),
                          "n_pooled_bins": len(edges) - 1})
    return pd.DataFrame(summaries), pd.DataFrame(correlations), pd.DataFrame(pairs)


def categorical_analysis(X, y, names):
    counts, associations, pairs = [], [], []
    for name in names:
        values = X[name].fillna("missing_or_unknown").astype(str)
        table = pd.crosstab(values, y).reindex(columns=range(5), fill_value=0)
        proportions = table / table.sum(axis=0)
        if table.shape[0] > 1:
            v = association(table.to_numpy(), method="cramer", correction=False)
            _, _, _, expected = chi2_contingency(table.to_numpy(), correction=False)
            sparse_fraction = float((expected < 5).mean())
        else:
            v, sparse_fraction = np.nan, np.nan
        associations.append({"feature": name, "cramers_v": v, "n": len(values),
                             "n_levels_including_unknown": len(table),
                             "n_missing": int(X[name].isna().sum()),
                             "fraction_expected_cells_below_5": sparse_fraction,
                             "estimator": "uncorrected Cramer V; descriptive only"})
        for level in table.index:
            for code in range(5):
                counts.append({"feature": name, "category": level, "class": f"C{code}",
                               "n": int(table.loc[level, code]),
                               "percentage_within_class": 100 * proportions.loc[level, code]})
        for a, b in PAIRS:
            pairs.append({"class_pair": f"C{a}-C{b}", "feature": name,
                          "categorical_overlap": float(np.minimum(proportions[a], proportions[b]).sum())})
    return pd.DataFrame(counts), pd.DataFrame(associations), pd.DataFrame(pairs)


def adjacent_summary(oof, numerical, categorical):
    matrix = confusion_matrix(oof.y_true, oof.y_pred, labels=range(5))
    supports = matrix.sum(axis=1)
    rows = []
    for a, b in PAIRS:
        pair = f"C{a}-C{b}"
        num = numerical[numerical.class_pair.eq(pair)]
        cat = categorical[categorical.class_pair.eq(pair)]
        forward, reverse = matrix[a, b] / supports[a], matrix[b, a] / supports[b]
        num_overlap = num.histogram_overlap.median()
        cat_overlap = cat.categorical_overlap.median()
        delta = num.abs_cliffs_delta.median()
        top = num.sort_values(["histogram_overlap", "feature"], ascending=[False, True]).head(3)
        representatives = "; ".join(f"{r.feature}: histogram OVL={r.histogram_overlap:.3f}" for r in top.itertuples())
        rows.append({"class_pair": pair, "n_lower": int(supports[a]), "n_higher": int(supports[b]),
                     "confusion_rate": (matrix[a, b] + matrix[b, a]) / (supports[a] + supports[b]),
                     "symmetric_confusion_rate": (forward + reverse) / 2,
                     "lower_to_higher_count": int(matrix[a, b]), "higher_to_lower_count": int(matrix[b, a]),
                     "lower_to_higher_rate": forward, "higher_to_lower_rate": reverse,
                     "median_numerical_histogram_overlap": num_overlap,
                     "median_categorical_overlap": cat_overlap,
                     "median_abs_cliffs_delta": delta,
                     "representative_feature_overlap": representatives,
                     "interpretation": f"Development OOF C{a}->C{b}: {forward:.1%}; C{b}->C{a}: {reverse:.1%}. "
                                       f"Median marginal numerical OVL={num_overlap:.3f}, categorical OVL={cat_overlap:.3f}, "
                                       f"median |Cliff delta|={delta:.3f}. Descriptive association; cannot apportion causality."})
    return pd.DataFrame(rows)


def plot_features(X, y, numerical, associations, adjacent):
    plt = granularity.pyplot()
    outputs = []
    fig, axes = plt.subplots(5, 3, figsize=(14, 17), layout="constrained")
    for ax, name in zip(axes.flat, numerical):
        boxes = ax.boxplot([X.loc[y.eq(i), name].dropna() for i in range(5)],
                           tick_labels=[f"C{i}" for i in range(5)], patch_artist=True,
                           flierprops={"markersize": 2, "alpha": 0.25},
                           medianprops={"color": "#222222", "linewidth": 1.6})
        for box, color in zip(boxes["boxes"], granularity.COLORS):
            box.set_facecolor(color)
            box.set_alpha(0.65)
        ax.set_title(name)
        ax.grid(axis="y", alpha=0.15)
    for ax in list(axes.flat)[len(numerical):]:
        ax.set_visible(False)
    fig.suptitle("Observed numerical features by original severity (development only)\nMedians, IQRs, 1.5-IQR whiskers and outliers; missing values excluded per feature", fontsize=13)
    outputs.append(granularity.save_figure(fig, "overlap_numerical_boxplots.png"))
    ordered = associations.sort_values("cramers_v")
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    ax.barh(ordered.feature, ordered.cramers_v, color=granularity.COLORS[0])
    ax.set(xlabel="Cramér's V (uncorrected; association, not causation)",
           title="Categorical F4 features vs C0-C4 on development")
    ax.grid(axis="x", alpha=0.15)
    outputs.append(granularity.save_figure(fig, "overlap_categorical_association.png"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    positions = np.arange(len(adjacent))
    axes[0].bar(positions - 0.18, adjacent.median_numerical_histogram_overlap, width=0.36,
                color=granularity.COLORS[0], label="Numerical histogram OVL")
    axes[0].bar(positions + 0.18, adjacent.median_categorical_overlap, width=0.36,
                color=granularity.COLORS[1], label="Categorical OVL")
    axes[0].set(xticks=positions, xticklabels=adjacent.class_pair, ylim=(0, 1),
                ylabel="Median marginal overlap", title="Distribution overlap (larger = more overlap)")
    axes[0].legend(fontsize=8)
    axes[1].bar(positions - 0.18, adjacent.lower_to_higher_rate, width=0.36,
                color=granularity.COLORS[2], label="Lower → higher")
    axes[1].bar(positions + 0.18, adjacent.higher_to_lower_rate, width=0.36,
                color=granularity.COLORS[4], label="Higher → lower")
    axes[1].set(xticks=positions, xticklabels=adjacent.class_pair, ylim=(0, 1),
                ylabel="Fraction of source class", title="Adjacent errors from T5 development OOF")
    axes[1].legend(fontsize=8)
    outputs.append(granularity.save_figure(fig, "overlap_adjacent_classes.png"))
    return outputs


def pca_analysis(X, y, inventory):
    """Descriptive full-development fit, not CV performance estimation."""
    preprocessor = frozen.ablation.build_preprocessor(X.columns.tolist(), inventory)
    transformed = preprocessor.fit_transform(X)
    pca = PCA(n_components=2, svd_solver="full")
    coordinates = pca.fit_transform(transformed)
    output = pd.DataFrame({"SEQN": X.index, "class": [f"C{i}" for i in y],
                           "PC1": coordinates[:, 0], "PC2": coordinates[:, 1]})
    variance = pd.DataFrame({"component": ["PC1", "PC2"],
                            "explained_variance_ratio": pca.explained_variance_ratio_,
                            "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_)})
    loadings = pd.DataFrame({"transformed_feature": preprocessor.get_feature_names_out(),
                            "PC1_loading": pca.components_[0], "PC2_loading": pca.components_[1]})
    plt = granularity.pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharex=True, sharey=True, layout="constrained")
    for code in range(5):
        mask = y.to_numpy() == code
        axes[0, 0].scatter(*coordinates[mask].T, s=12 if code == 0 else 20,
                           color=granularity.COLORS[code], alpha=0.2 if code == 0 else 0.65,
                           linewidths=0, label=f"C{code} (n={mask.sum()})")
        ax = list(axes.flat)[code + 1]
        ax.scatter(*coordinates.T, s=6, color="#bbbbbb", alpha=0.15, linewidths=0)
        ax.scatter(*coordinates[mask].T, s=16, color=granularity.COLORS[code], alpha=0.65, linewidths=0)
        ax.set_title(f"C{code} against all development records")
    axes[0, 0].set_title("All five classes")
    axes[0, 0].legend(fontsize=8, markerscale=1.5)
    for ax in axes.flat:
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    fig.suptitle("Exploratory PCA on development F4\nNumeric median/scaling + categorical imputation/one-hot; full-development EDA fit only")
    figure = granularity.save_figure(fig, "overlap_pca_C0_C4.png")
    return output, variance, loadings, figure, transformed.shape[1]


def main():
    if RUN_PATH.exists() or (TABLES_DIR / "adjacent_class_overlap.csv").exists():
        raise FileExistsError("Overlap output exists; refusing to overwrite")
    X, y, inventory, total, _, _, spec = granularity.load_development()
    protocol = granularity.freeze_protocol(X, spec)
    run = load_json(granularity.RUN_PATH)
    assert run["protocol_sha256"] == sha256_file(granularity.PROTOCOL_PATH)
    assert not run["heldout_used"] and run["T5_reproduces_prior_RF_OOF"]
    granularity.verify_hashes(run["outputs_sha256"])
    oof = pd.read_csv(TABLES_DIR / "granularity_oof_predictions.csv")
    oof = oof[oof.task.eq("T5")].set_index("SEQN").loc[X.index]
    np.testing.assert_array_equal(oof.y_true, y)
    numerical = inventory.loc[inventory.feature_type.eq("numerical"), "feature"].tolist()
    categorical = inventory.loc[inventory.feature_type.eq("categorical"), "feature"].tolist()
    assert len(numerical) == 15 and len(categorical) == 19
    summaries, correlations, numeric_pairs = numerical_analysis(X, y, total, numerical)
    counts, associations, cat_pairs = categorical_analysis(X, y, categorical)
    adjacent = adjacent_summary(oof, numeric_pairs, cat_pairs)
    outputs = []
    for frame, name in [(summaries, "numerical_summary"), (correlations, "spearman_correlations"),
                        (numeric_pairs, "numerical_adjacent_effects"), (counts, "categorical_crosstabs"),
                        (associations, "categorical_associations"), (cat_pairs, "categorical_adjacent_overlap")]:
        outputs.append(granularity.save_table(frame, f"overlap_{name}.csv"))
    outputs.append(granularity.save_table(adjacent, "adjacent_class_overlap.csv"))
    outputs.extend(plot_features(X, y, numerical, associations, adjacent))
    coordinates, variance, loadings, figure, n_transformed = pca_analysis(X, y, inventory)
    for frame, name in [(coordinates, "pca_coordinates"), (variance, "pca_variance"), (loadings, "pca_loadings")]:
        outputs.append(granularity.save_table(frame, f"overlap_{name}.csv"))
    outputs.append(figure)
    granularity.verify_hashes(protocol["preserved_files_sha256"])
    frozen.write_new_json({"completed_utc": frozen.now(), "development_n": len(X),
                           "heldout_used": False, "all_old_files_unchanged": True,
                           "n_raw_features": X.shape[1], "n_transformed_features_PCA": n_transformed,
                           "PCA_fit_scope": "full development, EDA only, separate from all CV estimators",
                           "PCA_explained_variance_ratio": variance.explained_variance_ratio.tolist(),
                           "limitations": ["Marginal overlap does not establish multivariate inseparability",
                                           "Histogram overlap depends on pooled decile bins and small-class sample size",
                                           "Cliff delta near zero does not imply identical distributions",
                                           "Cramer V is uncorrected; sparse expected cells are reported; no categorical p-value claim",
                                           "Spearman p-values are exploratory, unadjusted and not used for feature selection",
                                           "NHANES survey weights/PSUs/strata not modeled; sample-level descriptive results",
                                           "Complete-PHQ cohort and feature-specific missingness may induce selection bias",
                                           "Cannot separate causal effects of imbalance, overlap and label refinement",
                                           "PCA preserves only part of variance; projected mixing is not a separability test"],
                           "protocol_sha256": sha256_file(granularity.PROTOCOL_PATH),
                           "outputs_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in outputs}}, RUN_PATH)
    print(adjacent.to_string(index=False), flush=True)
    print("Numerical Spearman correlations:\n" + correlations.sort_values("abs_spearman_rho", ascending=False).to_string(index=False), flush=True)
    print("Categorical associations:\n" + associations.sort_values("cramers_v", ascending=False).to_string(index=False), flush=True)
    print("PCA variance:\n" + variance.to_string(index=False), flush=True)
    print("PASS: development-only overlap completed; old files preserved. Reports not edited.", flush=True)


if __name__ == "__main__":
    main()
