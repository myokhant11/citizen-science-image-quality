from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IN_PATH = Path("data/private/image_summary_base.csv")
OUT_DIR = Path("data/private/review_priority")
OUT_PATH = OUT_DIR / "image_review_priority.csv"
THRESHOLD_PATH = OUT_DIR / "quality_thresholds.csv"
FIG_DIR = OUT_DIR / "figures"

MODERATE_PERCENTILE = 0.75
HIGH_PERCENTILE = 0.90

METRICS = {
    "stomatal_index_iqr": "disagreement",
    "uncertain_mark_fraction": "uncertainty",
}

COLORS = {
    "green": "#2ca02c",
    "yellow": "#f2c744",
    "red": "#d62728",
}


def as_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)


def signal_level(value: float, moderate: float, high: float) -> str:
    if pd.isna(value):
        return "not_assessed"
    if value >= high:
        return "high"
    if value >= moderate:
        return "moderate"
    return "normal"


def review_reason(row: pd.Series) -> str:
    reasons = []

    if row["flag_insufficient_coverage"]:
        reasons.append("insufficient participant coverage")

    for label in ["disagreement", "uncertainty"]:
        level = row[f"{label}_level"]
        if level == "high":
            reasons.append(f"high {label}")
        elif level == "moderate":
            reasons.append(f"moderate {label}")

    return "; ".join(reasons) if reasons else "no unusual quality signal"


def assign_priority(row: pd.Series) -> str:
    if row["flag_insufficient_coverage"]:
        return "red"

    levels = [row["disagreement_level"], row["uncertainty_level"]]
    if "high" in levels:
        return "red"
    if levels.count("moderate") >= 2:
        return "red"
    if levels.count("moderate") == 1:
        return "yellow"
    return "green"


def create_figures(df: pd.DataFrame, thresholds: dict) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")

    eligible = df[df["analysis_eligible"]]
    fig, ax = plt.subplots(figsize=(9, 6))
    for priority in ["green", "yellow", "red"]:
        subset = eligible[eligible["review_priority"] == priority]
        ax.scatter(
            subset["stomatal_index_iqr"],
            subset["uncertain_mark_fraction"],
            label=f"{priority.title()} (n={len(subset)})",
            color=COLORS[priority],
            s=58,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.4,
        )

    disagreement = thresholds["stomatal_index_iqr"]
    uncertainty = thresholds["uncertain_mark_fraction"]
    ax.axvline(
        disagreement["moderate"], color="#777777", linestyle=":", linewidth=1
    )
    ax.axvline(
        disagreement["high"], color="#333333", linestyle="--", linewidth=1
    )
    ax.axhline(
        uncertainty["moderate"], color="#777777", linestyle=":", linewidth=1
    )
    ax.axhline(
        uncertainty["high"], color="#333333", linestyle="--", linewidth=1
    )
    ax.set_xlabel("Stomatal-index IQR (volunteer disagreement)")
    ax.set_ylabel("Fraction of marks labelled uncertain")
    ax.set_title("Relative review priority among adequately covered images")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "review_priority_scatter.png", dpi=150)
    plt.close(fig)

    counts = df["review_priority"].value_counts().reindex(
        ["green", "yellow", "red"], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        [label.title() for label in counts.index],
        counts.values,
        color=[COLORS[label] for label in counts.index],
        edgecolor="black",
    )
    ax.bar_label(bars)
    ax.set_ylabel("Number of images")
    ax.set_title("Images by review priority")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "review_priority_counts.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_PATH)
    df["analysis_eligible"] = as_boolean(df["analysis_eligible"])
    df["flag_insufficient_coverage"] = as_boolean(
        df["flag_insufficient_coverage"]
    )

    eligible = df[df["analysis_eligible"]].copy()
    if eligible.empty:
        raise ValueError("No analysis-eligible images are available.")

    thresholds = {}
    threshold_rows = []
    for metric, label in METRICS.items():
        moderate = float(eligible[metric].quantile(MODERATE_PERCENTILE))
        high = float(eligible[metric].quantile(HIGH_PERCENTILE))
        thresholds[metric] = {"moderate": moderate, "high": high}
        threshold_rows.append({
            "metric": metric,
            "quality_signal": label,
            "moderate_percentile": int(MODERATE_PERCENTILE * 100),
            "moderate_threshold": moderate,
            "high_percentile": int(HIGH_PERCENTILE * 100),
            "high_threshold": high,
        })

        level_column = f"{label}_level"
        df[level_column] = "not_assessed"
        assessed = df["analysis_eligible"]
        df.loc[assessed, level_column] = df.loc[assessed, metric].apply(
            signal_level,
            args=(moderate, high),
        )

        percentile_column = f"{label}_percentile"
        df[percentile_column] = np.nan
        df.loc[assessed, percentile_column] = (
            df.loc[assessed, metric].rank(method="average", pct=True) * 100
        )

    df["review_priority"] = df.apply(assign_priority, axis=1)
    df["review_reason"] = df.apply(review_reason, axis=1)
    df["review_score_percentile"] = df[[
        "disagreement_percentile",
        "uncertainty_percentile",
    ]].max(axis=1)
    df.loc[df["flag_insufficient_coverage"], "review_score_percentile"] = 100.0

    priority_order = {"red": 0, "yellow": 1, "green": 2}
    df["priority_sort"] = df["review_priority"].map(priority_order)
    df = df.sort_values(
        ["priority_sort", "review_score_percentile", "filename_normalized"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    df["review_rank"] = np.arange(1, len(df) + 1)
    df = df.drop(columns="priority_sort")

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(THRESHOLD_PATH, index=False)
    df.to_csv(OUT_PATH, index=False)
    create_figures(df, thresholds)

    print("=== RELATIVE THRESHOLDS ===")
    for metric, label in METRICS.items():
        values = thresholds[metric]
        print(f"{label.title():12} moderate (75th): {values['moderate']:.6f}")
        print(f"{label.title():12} high     (90th): {values['high']:.6f}")

    print("\n=== REVIEW PRIORITY ===")
    counts = df["review_priority"].value_counts()
    for priority in ["green", "yellow", "red"]:
        print(f"{priority.title():8}: {int(counts.get(priority, 0))}")

    print("\n=== TOP 10 REVIEW QUEUE ===")
    print(df.head(10)[[
        "review_rank",
        "filename_normalized",
        "review_priority",
        "review_score_percentile",
        "review_reason",
    ]].to_string(index=False))

    print("\nInterpretation: priority and score describe relative review need,")
    print("not the probability that an image is scientifically incorrect.")
    print(f"\nWrote: {OUT_PATH.resolve()}")
    print(f"Wrote: {THRESHOLD_PATH.resolve()}")
    print(f"Figures: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
