from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IN_PATH = Path("data/private/classifications_clean.csv")
OUT_DIR = Path("data/private/quality_profile")
FIG_DIR = OUT_DIR / "figures"

COUNT_COLUMNS = [
    "count_stoma",
    "count_regular_cell",
    "count_not_sure",
    "count_unclear_patch",
]

BOOLEAN_COLUMNS = [
    "annotation_parse_ok",
    "subject_data_parse_ok",
    "subject_id_matches_subject_data",
    "has_multiple_subject_records",
    "has_duplicate_task",
    "is_empty",
    "is_repeat_group",
    "is_additional_submission",
    "t4_present",
]


def as_boolean(series: pd.Series) -> pd.Series:
    """Read boolean columns consistently across pandas versions."""
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)


def iqr(series: pd.Series) -> float:
    """Interquartile range: the middle 50% spread."""
    clean = series.dropna()
    if clean.empty:
        return np.nan
    return float(clean.quantile(0.75) - clean.quantile(0.25))


def mad(series: pd.Series) -> float:
    """Median absolute deviation: a robust measure of disagreement."""
    clean = series.dropna()
    if clean.empty:
        return np.nan
    median = clean.median()
    return float((clean - median).abs().median())


def pooled_stomatal_index(frame: pd.DataFrame) -> float:
    """Calculate SI from the combined stomata and regular-cell counts."""
    stoma = frame["count_stoma"].sum()
    regular = frame["count_regular_cell"].sum()
    denominator = stoma + regular
    return float(stoma / denominator) if denominator > 0 else np.nan


def build_repeat_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Describe how repeated submissions from the same participant differ."""
    keys = ["participant_id", "filename_normalized"]
    repeated = df[df["is_repeat_group"]].copy()

    grouped = repeated.groupby(keys, sort=True)
    profile = grouped.agg(
        n_submissions=("classification_id", "size"),
        contains_empty=("is_empty", "any"),
        stoma_min=("count_stoma", "min"),
        stoma_max=("count_stoma", "max"),
        regular_min=("count_regular_cell", "min"),
        regular_max=("count_regular_cell", "max"),
        not_sure_min=("count_not_sure", "min"),
        not_sure_max=("count_not_sure", "max"),
        unclear_min=("count_unclear_patch", "min"),
        unclear_max=("count_unclear_patch", "max"),
        si_min=("stomatal_index", "min"),
        si_max=("stomatal_index", "max"),
    )

    unique_vectors = grouped[COUNT_COLUMNS].apply(
        lambda group: len(group.drop_duplicates())
    )
    profile["n_unique_count_vectors"] = unique_vectors
    profile["all_counts_identical"] = profile["n_unique_count_vectors"] == 1
    profile["stoma_range"] = profile["stoma_max"] - profile["stoma_min"]
    profile["regular_range"] = profile["regular_max"] - profile["regular_min"]
    profile["not_sure_range"] = profile["not_sure_max"] - profile["not_sure_min"]
    profile["unclear_range"] = profile["unclear_max"] - profile["unclear_min"]
    profile["si_range"] = profile["si_max"] - profile["si_min"]
    return profile.reset_index()


def build_image_profile(eligible: pd.DataFrame) -> pd.DataFrame:
    """Create image-level diagnostics without removing repeat submissions."""
    image_key = "filename_normalized"
    grouped = eligible.groupby(image_key, sort=True)

    image = grouped.agg(
        n_classifications_raw=("classification_id", "size"),
        n_unique_participants=("participant_id", "nunique"),
        n_subject_ids=("subject_id", "nunique"),
        n_original_filenames=("filename_original", "nunique"),
        n_additional_submissions=("is_additional_submission", "sum"),
        n_duplicate_task_rows=("has_duplicate_task", "sum"),
        t1_no_count=("t1_is_no", "sum"),
        t1_missing_count=("t1_is_missing", "sum"),
        t4_count=("t4_present", "sum"),
        stoma_mean_raw=("count_stoma", "mean"),
        stoma_median_raw=("count_stoma", "median"),
        stoma_std_raw=("count_stoma", "std"),
        stoma_min_raw=("count_stoma", "min"),
        stoma_max_raw=("count_stoma", "max"),
        regular_mean_raw=("count_regular_cell", "mean"),
        regular_median_raw=("count_regular_cell", "median"),
        not_sure_mean=("count_not_sure", "mean"),
        unclear_mean=("count_unclear_patch", "mean"),
        uncertain_marks_sum=("uncertain_marks", "sum"),
        total_marks_sum=("total_marks", "sum"),
        n_old_rule_outliers=("old_rule_outlier", "sum"),
    )

    image["source_filenames"] = grouped["filename_original"].apply(
        lambda values: " | ".join(sorted(set(values)))
    )
    image["stoma_iqr_raw"] = grouped["count_stoma"].apply(iqr)
    image["stoma_mad_raw"] = grouped["count_stoma"].apply(mad)
    image["regular_iqr_raw"] = grouped["count_regular_cell"].apply(iqr)
    image["stomatal_index_iqr_raw"] = grouped["stomatal_index"].apply(iqr)
    image["si_pooled_raw"] = grouped.apply(
        pooled_stomatal_index, include_groups=False
    )
    image["uncertain_mark_fraction"] = (
        image["uncertain_marks_sum"] / image["total_marks_sum"].replace(0, np.nan)
    )
    image["old_rule_outlier_fraction"] = (
        image["n_old_rule_outliers"] / image["n_classifications_raw"]
    )

    # Give each participant one contribution per normalized image. If a person
    # submitted repeatedly, their median counts form that one contribution.
    participant_image = eligible.groupby(
        ["participant_id", image_key], as_index=False, sort=True
    ).agg(
        count_stoma=("count_stoma", "median"),
        count_regular_cell=("count_regular_cell", "median"),
        count_not_sure=("count_not_sure", "median"),
        count_unclear_patch=("count_unclear_patch", "median"),
        n_submissions=("classification_id", "size"),
    )
    participant_image["total_cells"] = (
        participant_image["count_stoma"]
        + participant_image["count_regular_cell"]
    )
    participant_image["stomatal_index"] = np.where(
        participant_image["total_cells"] > 0,
        participant_image["count_stoma"] / participant_image["total_cells"],
        np.nan,
    )

    participant_grouped = participant_image.groupby(image_key, sort=True)
    participant_summary = participant_grouped.agg(
        n_participant_contributions=("participant_id", "size"),
        stoma_median_participant=("count_stoma", "median"),
        regular_median_participant=("count_regular_cell", "median"),
    )
    participant_summary["stoma_iqr_participant"] = participant_grouped[
        "count_stoma"
    ].apply(iqr)
    participant_summary["stoma_mad_participant"] = participant_grouped[
        "count_stoma"
    ].apply(mad)
    participant_summary["si_pooled_participant_collapsed"] = (
        participant_grouped.apply(pooled_stomatal_index, include_groups=False)
    )

    # Reproduce the old stoma-only rule for diagnosis, not as a decision.
    old_filtered = eligible[~eligible["old_rule_outlier"]]
    old_filtered_si = old_filtered.groupby(image_key, sort=True).apply(
        pooled_stomatal_index, include_groups=False
    )
    old_filtered_si.name = "si_pooled_old_rule_filtered"

    image = image.join(participant_summary).join(old_filtered_si)
    image["abs_si_change_after_collapsing_repeats"] = (
        image["si_pooled_participant_collapsed"] - image["si_pooled_raw"]
    ).abs()
    image["abs_si_change_old_rule"] = (
        image["si_pooled_old_rule_filtered"] - image["si_pooled_raw"]
    ).abs()
    return image.reset_index()


def create_figures(image: pd.DataFrame, repeat: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(image["n_classifications_raw"], bins=20, edgecolor="black", alpha=0.8)
    ax.set_xlabel("Classifications per normalized image")
    ax.set_ylabel("Number of images")
    ax.set_title("Volunteer coverage across images")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "classifications_per_image.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sizes = 20 + 1.2 * image["n_unique_participants"].clip(upper=100)
    scatter = ax.scatter(
        image["stoma_iqr_participant"],
        image["uncertain_mark_fraction"],
        s=sizes,
        c=image["n_unique_participants"],
        cmap="viridis",
        alpha=0.75,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlabel("Stoma-count IQR after one contribution per participant")
    ax.set_ylabel("Fraction of marks labelled uncertain")
    ax.set_title("Disagreement and uncertainty by image")
    fig.colorbar(scatter, ax=ax, label="Unique participants")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "disagreement_vs_uncertainty.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    visible_limit = max(5, int(np.ceil(repeat["stoma_range"].quantile(0.99))))
    visible = repeat.loc[repeat["stoma_range"] <= visible_limit, "stoma_range"]
    ax.hist(visible, bins=range(0, visible_limit + 2), edgecolor="black", alpha=0.8)
    ax.set_xlabel("Range of stoma counts within a repeat group")
    ax.set_ylabel("Participant-image repeat groups")
    ax.set_title("Consistency of repeated submissions (up to 99th percentile)")
    ax.text(
        0.98,
        0.95,
        f"maximum observed range = {repeat['stoma_range'].max():.0f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "repeat_stoma_range.png", dpi=150)
    plt.close(fig)

    comparison = image[image["n_unique_participants"] >= 5]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        comparison["si_pooled_raw"],
        comparison["si_pooled_participant_collapsed"],
        s=42,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.4,
    )
    low = min(comparison["si_pooled_raw"].min(), comparison["si_pooled_participant_collapsed"].min())
    high = max(comparison["si_pooled_raw"].max(), comparison["si_pooled_participant_collapsed"].max())
    ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Pooled SI using every submission")
    ax.set_ylabel("Pooled SI after one contribution per participant")
    ax.set_title("Effect of repeated submissions (images with at least 5 participants)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "repeat_submission_sensitivity.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        IN_PATH,
        dtype={
            "classification_id": str,
            "participant_id": str,
            "subject_id": str,
            "subject_data_subject_id": str,
        },
        keep_default_na=False,
    )

    missing = sorted(
        set(COUNT_COLUMNS + BOOLEAN_COLUMNS + [
            "classification_id",
            "participant_id",
            "subject_id",
            "filename_original",
            "filename_normalized",
            "t1_answer",
            "total_marks",
        ]) - set(df.columns)
    )
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    for column in COUNT_COLUMNS + ["total_marks"]:
        df[column] = pd.to_numeric(df[column], errors="raise")
    for column in BOOLEAN_COLUMNS:
        df[column] = as_boolean(df[column])

    df["total_cells"] = df["count_stoma"] + df["count_regular_cell"]
    df["stomatal_index"] = np.where(
        df["total_cells"] > 0,
        df["count_stoma"] / df["total_cells"],
        np.nan,
    )

    base_valid = (
        df["annotation_parse_ok"]
        & df["subject_data_parse_ok"]
        & df["subject_id_matches_subject_data"]
        & ~df["has_multiple_subject_records"]
        & ~df["is_empty"]
    )
    eligible = df[base_valid].copy()
    eligible["uncertain_marks"] = (
        eligible["count_not_sure"] + eligible["count_unclear_patch"]
    )
    eligible["t1_is_no"] = eligible["t1_answer"].eq("No")
    eligible["t1_is_missing"] = eligible["t1_answer"].eq("")

    image_median_stoma = eligible.groupby("filename_normalized")[
        "count_stoma"
    ].transform("median")
    eligible["old_rule_outlier"] = (
        (eligible["count_stoma"] < 0.3 * image_median_stoma)
        | (eligible["count_stoma"] > 3.0 * image_median_stoma)
    )

    repeat_profile = build_repeat_profile(df)
    image_profile = build_image_profile(eligible)

    repeat_path = OUT_DIR / "repeat_group_profile.csv"
    image_path = OUT_DIR / "image_quality_profile.csv"
    repeat_profile.to_csv(repeat_path, index=False)
    image_profile.to_csv(image_path, index=False)
    create_figures(image_profile, repeat_profile)

    print("=== DATA BASIS ===")
    print(f"All classifications:                {len(df)}")
    print(f"Eligible non-empty classifications: {len(eligible)}")
    print(f"Normalized images:                  {df['filename_normalized'].nunique()}")
    print(f"Unique participants:                {df['participant_id'].nunique()}")

    print("\n=== REPEATED SUBMISSIONS ===")
    print(f"Participant-image repeat groups:    {len(repeat_profile)}")
    print(f"Rows in repeat groups:              {int(df['is_repeat_group'].sum())}")
    print(f"Additional submissions:             {int(df['is_additional_submission'].sum())}")
    print(f"Groups with identical count vectors:{int(repeat_profile['all_counts_identical'].sum()):>5}")
    print(f"Median stoma range in repeat groups: {repeat_profile['stoma_range'].median():.2f}")
    print(f"Maximum stoma range:                 {repeat_profile['stoma_range'].max():.0f}")

    print("\n=== T1 AND T4 STRUCTURE ===")
    display_t1 = df["t1_answer"].replace("", "(missing)")
    print(pd.crosstab(display_t1, df["t4_present"], dropna=False).to_string())

    print("\n=== OLD STOMA-ONLY OUTLIER RULE: DIAGNOSTIC ONLY ===")
    print(f"Rows flagged:                        {int(eligible['old_rule_outlier'].sum())}")
    print(f"Images affected:                     {(image_profile['n_old_rule_outliers'] > 0).sum()}")
    print(f"Median absolute SI change:           {image_profile['abs_si_change_old_rule'].median():.6f}")
    print(f"Maximum absolute SI change:          {image_profile['abs_si_change_old_rule'].max():.6f}")

    print("\n=== EFFECT OF COLLAPSING REPEATS ===")
    repeat_images = image_profile[image_profile["n_additional_submissions"] > 0]
    print(f"Images containing repeat submissions:{len(repeat_images):>5}")
    print(f"Median absolute SI change:           {image_profile['abs_si_change_after_collapsing_repeats'].median():.6f}")
    print(f"Median change among affected images: {repeat_images['abs_si_change_after_collapsing_repeats'].median():.6f}")
    print(f"Maximum absolute SI change:          {image_profile['abs_si_change_after_collapsing_repeats'].max():.6f}")
    print(f"Images with change > 0.001:          {(image_profile['abs_si_change_after_collapsing_repeats'] > 0.001).sum()}")

    print("\n=== LOWEST PARTICIPANT COVERAGE ===")
    lowest = image_profile.nsmallest(5, "n_unique_participants")[[
        "filename_normalized",
        "n_classifications_raw",
        "n_unique_participants",
        "n_additional_submissions",
    ]]
    print(lowest.to_string(index=False))

    print(f"\nWrote: {image_path.resolve()}")
    print(f"Wrote: {repeat_path.resolve()}")
    print(f"Figures: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
