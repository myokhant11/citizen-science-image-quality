from pathlib import Path

import numpy as np
import pandas as pd


IN_PATH = Path("data/private/classifications_clean.csv")
PARTICIPANT_OUT = Path("data/private/participant_image_first.csv")
IMAGE_OUT = Path("data/private/image_summary_base.csv")

# This is an operational minimum, not a claim that five people guarantee
# scientific accuracy. In this dataset it separates the single one-person
# image from all other images, which have at least 42 participants.
MIN_PARTICIPANTS = 5

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
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)


def iqr(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    return float(clean.quantile(0.75) - clean.quantile(0.25))


def mad(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    median = clean.median()
    return float((clean - median).abs().median())


def pooled_stomatal_index(frame: pd.DataFrame) -> float:
    stoma = frame["count_stoma"].sum()
    regular = frame["count_regular_cell"].sum()
    denominator = stoma + regular
    return float(stoma / denominator) if denominator > 0 else np.nan


def main() -> None:
    PARTICIPANT_OUT.parent.mkdir(parents=True, exist_ok=True)

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

    for column in COUNT_COLUMNS + ["total_marks"]:
        df[column] = pd.to_numeric(df[column], errors="raise")
    for column in BOOLEAN_COLUMNS:
        df[column] = as_boolean(df[column])

    df["row_order"] = np.arange(len(df))
    df["created_at_utc"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["structural_valid"] = (
        df["annotation_parse_ok"]
        & df["subject_data_parse_ok"]
        & df["subject_id_matches_subject_data"]
        & ~df["has_multiple_subject_records"]
    )
    df["eligible_submission"] = df["structural_valid"] & ~df["is_empty"]

    eligible = df[df["eligible_submission"]].copy()
    eligible["total_cells"] = (
        eligible["count_stoma"] + eligible["count_regular_cell"]
    )
    eligible["stomatal_index"] = np.where(
        eligible["total_cells"] > 0,
        eligible["count_stoma"] / eligible["total_cells"],
        np.nan,
    )
    eligible["uncertain_marks"] = (
        eligible["count_not_sure"] + eligible["count_unclear_patch"]
    )
    eligible["t1_is_no"] = eligible["t1_answer"].eq("No")
    eligible["t1_is_missing"] = eligible["t1_answer"].eq("")

    # Repeated submissions are not independent votes. Select the first valid,
    # non-empty submission so every participant contributes exactly once. This
    # keeps one real, internally coherent count vector and avoids creating a
    # synthetic median vector when only two conflicting submissions exist.
    keys = ["participant_id", "filename_normalized"]
    eligible = eligible.sort_values(
        keys + ["created_at_utc", "row_order"],
        na_position="last",
    )
    eligible["valid_submissions_in_group"] = eligible.groupby(keys)[
        "classification_id"
    ].transform("size")
    contribution = eligible.drop_duplicates(keys, keep="first").copy()

    participant_columns = [
        "participant_id",
        "filename_normalized",
        "classification_id",
        "created_at",
        "subject_id",
        "filename_original",
        *COUNT_COLUMNS,
        "total_marks",
        "total_cells",
        "stomatal_index",
        "uncertain_marks",
        "t1_answer",
        "t4_present",
        "has_duplicate_task",
        "valid_submissions_in_group",
    ]
    participant_image = contribution[participant_columns].copy()
    participant_image.to_csv(PARTICIPANT_OUT, index=False)

    image_key = "filename_normalized"

    # Audit counts use every raw row, including rows that are later only flagged.
    raw_grouped = df.groupby(image_key, sort=True)
    audit = raw_grouped.agg(
        n_classifications_raw=("classification_id", "size"),
        n_empty_classifications=("is_empty", "sum"),
        n_structurally_invalid=("structural_valid", lambda values: int((~values).sum())),
        n_duplicate_task_rows=("has_duplicate_task", "sum"),
        n_subject_ids=("subject_id", "nunique"),
        n_original_filenames=("filename_original", "nunique"),
    )
    audit["source_filenames"] = raw_grouped["filename_original"].apply(
        lambda values: " | ".join(sorted(set(values)))
    )

    # Analytical metrics use one selected contribution per participant and image.
    grouped = participant_image.groupby(image_key, sort=True)
    summary = grouped.agg(
        n_unique_participants=("participant_id", "size"),
        n_participants_with_repeats=(
            "valid_submissions_in_group",
            lambda values: int((values > 1).sum()),
        ),
        n_additional_submissions_ignored=(
            "valid_submissions_in_group",
            lambda values: int((values - 1).sum()),
        ),
        max_submissions_one_participant=("valid_submissions_in_group", "max"),
        stoma_mean=("count_stoma", "mean"),
        stoma_median=("count_stoma", "median"),
        stoma_std=("count_stoma", "std"),
        stoma_min=("count_stoma", "min"),
        stoma_max=("count_stoma", "max"),
        regular_mean=("count_regular_cell", "mean"),
        regular_median=("count_regular_cell", "median"),
        regular_std=("count_regular_cell", "std"),
        not_sure_mean=("count_not_sure", "mean"),
        unclear_mean=("count_unclear_patch", "mean"),
        uncertain_marks_sum=("uncertain_marks", "sum"),
        total_marks_sum=("total_marks", "sum"),
        t1_no_count=("t1_answer", lambda values: int((values == "No").sum())),
        t1_missing_count=("t1_answer", lambda values: int((values == "").sum())),
        t4_count=("t4_present", "sum"),
        stomatal_index_median=("stomatal_index", "median"),
        stomatal_index_mean=("stomatal_index", "mean"),
    )
    summary["stoma_iqr"] = grouped["count_stoma"].apply(iqr)
    summary["stoma_mad"] = grouped["count_stoma"].apply(mad)
    summary["regular_iqr"] = grouped["count_regular_cell"].apply(iqr)
    summary["regular_mad"] = grouped["count_regular_cell"].apply(mad)
    summary["stomatal_index_iqr"] = grouped["stomatal_index"].apply(iqr)
    summary["stomatal_index_mad"] = grouped["stomatal_index"].apply(mad)
    summary["stomatal_index_pooled"] = grouped.apply(
        pooled_stomatal_index,
        include_groups=False,
    )
    summary["uncertain_mark_fraction"] = (
        summary["uncertain_marks_sum"]
        / summary["total_marks_sum"].replace(0, np.nan)
    )
    summary["t1_no_fraction"] = (
        summary["t1_no_count"] / summary["n_unique_participants"]
    )
    summary["repeat_participant_fraction"] = (
        summary["n_participants_with_repeats"]
        / summary["n_unique_participants"]
    )

    summary = audit.join(summary).reset_index()
    summary["flag_insufficient_coverage"] = (
        summary["n_unique_participants"] < MIN_PARTICIPANTS
    )
    summary["analysis_eligible"] = ~summary["flag_insufficient_coverage"]

    assert len(participant_image) == len(
        participant_image.drop_duplicates(keys)
    ), "Participant-image contributions are not unique."
    assert len(eligible) - len(participant_image) == int(
        summary["n_additional_submissions_ignored"].sum()
    ), "Repeat-submission accounting does not balance."
    assert set(summary[image_key]) == set(df[image_key]), "An image was lost."

    summary.to_csv(IMAGE_OUT, index=False)

    print("=== PARTICIPANT-LEVEL CONSOLIDATION ===")
    print(f"Raw classifications:                 {len(df)}")
    print(f"Valid non-empty submissions:         {len(eligible)}")
    print(f"One-per-participant contributions:   {len(participant_image)}")
    print(f"Additional submissions ignored:      {len(eligible) - len(participant_image)}")

    print("\n=== IMAGE SUMMARY ===")
    print(f"Normalized images retained:          {len(summary)}")
    print(f"Images eligible for analysis:        {int(summary['analysis_eligible'].sum())}")
    print(f"Images with insufficient coverage:   {int(summary['flag_insufficient_coverage'].sum())}")
    print(f"Minimum participants, eligible set:  {int(summary.loc[summary['analysis_eligible'], 'n_unique_participants'].min())}")
    print(f"Maximum participants:                {int(summary['n_unique_participants'].max())}")

    print("\n=== INSUFFICIENT-COVERAGE IMAGE ===")
    low = summary[summary["flag_insufficient_coverage"]][[
        "filename_normalized",
        "n_classifications_raw",
        "n_unique_participants",
        "stomatal_index_median",
    ]]
    print(low.to_string(index=False))

    print(f"\nWrote: {PARTICIPANT_OUT.resolve()}")
    print(f"Wrote: {IMAGE_OUT.resolve()}")


if __name__ == "__main__":
    main()
