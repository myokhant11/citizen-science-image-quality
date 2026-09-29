from contextlib import closing
from pathlib import Path
import re
import sqlite3

import pandas as pd


DB_PATH = Path("data/private/zooniverse_quality.db")
OUTPUT_PATH = Path("data/public/image_quality_dashboard.csv")
TEMP_OUTPUT_PATH = Path("data/public/image_quality_dashboard.tmp.csv")


DASHBOARD_QUERY = """
SELECT
    filename_normalized AS image_id,
    review_priority,
    review_rank,
    review_score_percentile,
    review_reason,
    analysis_eligible,
    flag_insufficient_coverage,
    n_classifications_raw,
    n_unique_participants,
    n_additional_submissions_ignored,
    stomatal_index_median,
    stomatal_index_pooled,
    stomatal_index_iqr,
    stoma_median,
    stoma_iqr,
    regular_median,
    regular_iqr,
    uncertain_mark_fraction,
    disagreement_level,
    disagreement_percentile,
    uncertainty_level,
    uncertainty_percentile
FROM image_review_priority
ORDER BY review_rank, filename_normalized
"""


def source_group(image_id: str) -> str:
    """Return a neutral source label derived only from the filename prefix."""
    prefixes = {
        "balt_": "BALT",
        "blandy_": "Blandy",
        "capitol_": "Capitol",
        "nmnh_": "NMNH",
        "usnah_": "USNAH",
        "rsb_": "RSB",
    }
    for prefix, label in prefixes.items():
        if image_id.startswith(prefix):
            return label
    return "Other"


def specimen_id(image_id: str) -> str:
    """Extract the RSB number when the filename contains one."""
    match = re.search(r"(?:^|_)rsb_?(\d+)(?:_|$)", image_id)
    return match.group(1) if match else ""


def spot_number(image_id: str):
    """Extract the image spot number for optional dashboard filtering."""
    match = re.search(r"_spot0*(\d+)(?:_|$)", image_id)
    return int(match.group(1)) if match else pd.NA


def read_dashboard_data() -> pd.DataFrame:
    """Read only the aggregated image-level table from SQLite."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH.resolve()}\n"
            "Run src/05_build_database.py first."
        )

    with closing(sqlite3.connect(DB_PATH)) as connection:
        frame = pd.read_sql_query(DASHBOARD_QUERY, connection)

    priority_order = {"red": 1, "yellow": 2, "green": 3}
    priority_colors = {
        "red": "#D62728",
        "yellow": "#F2C744",
        "green": "#2CA02C",
    }

    frame.insert(1, "source_group", frame["image_id"].map(source_group))
    frame.insert(2, "specimen_id", frame["image_id"].map(specimen_id))
    frame.insert(3, "spot_number", frame["image_id"].map(spot_number))
    frame.insert(
        5,
        "review_priority_order",
        frame["review_priority"].map(priority_order),
    )
    frame.insert(
        6,
        "priority_color_hex",
        frame["review_priority"].map(priority_colors),
    )
    frame.insert(
        11,
        "analysis_status",
        frame["analysis_eligible"].map(
            {1: "Eligible", 0: "Insufficient coverage"}
        ),
    )

    frame["spot_number"] = frame["spot_number"].astype("Int64")
    return frame


def validate_dashboard_data(frame: pd.DataFrame) -> None:
    """Check grain, privacy and fields required by the dashboard."""
    if frame.empty:
        raise ValueError("Dashboard export contains no images.")
    if frame["image_id"].duplicated().any():
        raise ValueError("Dashboard export must contain one row per image.")

    prohibited_columns = {
        "participant_id",
        "classification_id",
        "user_name",
        "user_id",
        "user_ip",
    }
    exposed = sorted(prohibited_columns.intersection(frame.columns))
    if exposed:
        raise ValueError(f"Dashboard export exposes private columns: {exposed}")

    expected_priorities = {"green", "yellow", "red"}
    actual_priorities = set(frame["review_priority"].dropna())
    if actual_priorities != expected_priorities:
        raise ValueError(
            "Unexpected review-priority values: "
            f"{sorted(actual_priorities)}"
        )

    required_complete = [
        "image_id",
        "review_priority",
        "review_priority_order",
        "priority_color_hex",
        "review_rank",
        "review_reason",
        "analysis_status",
    ]
    missing_values = frame[required_complete].isna().sum()
    if missing_values.any():
        raise ValueError(
            "Required dashboard fields contain missing values:\n"
            f"{missing_values[missing_values > 0].to_string()}"
        )


def write_dashboard_data(frame: pd.DataFrame) -> None:
    """Write the validated export, replacing an older generated version."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TEMP_OUTPUT_PATH.exists():
        TEMP_OUTPUT_PATH.unlink()

    frame.to_csv(TEMP_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    TEMP_OUTPUT_PATH.replace(OUTPUT_PATH)


def print_summary(frame: pd.DataFrame) -> None:
    priority_counts = (
        frame.groupby(
            ["review_priority_order", "review_priority"],
            as_index=False,
        )
        .size()
        .sort_values("review_priority_order")
        .drop(columns="review_priority_order")
        .rename(columns={"size": "image_count"})
    )
    source_counts = (
        frame.groupby("source_group", as_index=False)
        .size()
        .sort_values(["size", "source_group"], ascending=[False, True])
        .rename(columns={"size": "image_count"})
    )

    print("=== DASHBOARD EXPORT ===")
    print(f"Image rows:                 {len(frame)}")
    print(f"Unique images:              {frame['image_id'].nunique()}")
    print(f"Columns:                    {len(frame.columns)}")
    print("Participant-level records:  0")

    print("\n=== REVIEW PRIORITY ===")
    print(priority_counts.to_string(index=False))

    print("\n=== SOURCE GROUPS ===")
    print(source_counts.to_string(index=False))

    print(f"\nWrote: {OUTPUT_PATH.resolve()}")


def main() -> None:
    dashboard = read_dashboard_data()
    validate_dashboard_data(dashboard)
    write_dashboard_data(dashboard)
    print_summary(dashboard)


if __name__ == "__main__":
    main()
