from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from contextlib import closing

import pandas as pd


CLASSIFICATIONS_PATH = Path("data/private/classifications_clean.csv")
PARTICIPANT_IMAGE_PATH = Path("data/private/participant_image_first.csv")
IMAGE_REVIEW_PATH = Path(
    "data/private/review_priority/image_review_priority.csv"
)
THRESHOLDS_PATH = Path(
    "data/private/review_priority/quality_thresholds.csv"
)

DB_PATH = Path("data/private/zooniverse_quality.db")
TEMP_DB_PATH = Path("data/private/zooniverse_quality.tmp.db")


def read_inputs() -> dict[str, pd.DataFrame]:
    """Read the four validated pipeline outputs that form the database."""
    return {
        "classifications_clean": pd.read_csv(
            CLASSIFICATIONS_PATH,
            dtype={
                "classification_id": str,
                "participant_id": str,
                "subject_id": str,
                "subject_data_subject_id": str,
            },
            keep_default_na=False,
        ),
        "participant_image_contributions": pd.read_csv(
            PARTICIPANT_IMAGE_PATH,
            dtype={
                "classification_id": str,
                "participant_id": str,
                "subject_id": str,
            },
            keep_default_na=False,
        ),
        "image_review_priority": pd.read_csv(IMAGE_REVIEW_PATH),
        "quality_thresholds": pd.read_csv(THRESHOLDS_PATH),
    }


def validate_inputs(tables: dict[str, pd.DataFrame]) -> None:
    """Stop before database creation if keys or privacy rules are violated."""
    classifications = tables["classifications_clean"]
    contributions = tables["participant_image_contributions"]
    images = tables["image_review_priority"]

    direct_identifiers = {"user_name", "user_id", "user_ip"}
    for table_name, frame in tables.items():
        exposed = sorted(direct_identifiers.intersection(frame.columns))
        if exposed:
            raise ValueError(
                f"{table_name} contains prohibited identifier columns: {exposed}"
            )

    if classifications["classification_id"].duplicated().any():
        raise ValueError("classification_id is not unique.")
    if contributions.duplicated(
        ["participant_id", "filename_normalized"]
    ).any():
        raise ValueError("Participant-image contributions are not unique.")
    if images["filename_normalized"].duplicated().any():
        raise ValueError("Image identifiers are not unique.")

    contribution_images = set(contributions["filename_normalized"])
    image_table_images = set(images["filename_normalized"])
    if contribution_images != image_table_images:
        raise ValueError("Image keys do not match across pipeline outputs.")


def create_indexes_and_views(connection: sqlite3.Connection) -> None:
    """Add integrity constraints, query indexes and dashboard-ready views."""
    connection.executescript(
        """
        CREATE UNIQUE INDEX ux_classification_id
            ON classifications_clean(classification_id);

        CREATE INDEX ix_classification_image
            ON classifications_clean(filename_normalized);

        CREATE INDEX ix_classification_participant_image
            ON classifications_clean(participant_id, filename_normalized);

        CREATE UNIQUE INDEX ux_participant_image
            ON participant_image_contributions(
                participant_id,
                filename_normalized
            );

        CREATE INDEX ix_contribution_image
            ON participant_image_contributions(filename_normalized);

        CREATE UNIQUE INDEX ux_image_review
            ON image_review_priority(filename_normalized);

        CREATE INDEX ix_review_queue
            ON image_review_priority(review_priority, review_rank);

        CREATE VIEW v_priority_summary AS
        SELECT
            review_priority,
            COUNT(*) AS image_count,
            ROUND(100.0 * COUNT(*) /
                (SELECT COUNT(*) FROM image_review_priority), 1
            ) AS percentage
        FROM image_review_priority
        GROUP BY review_priority;

        CREATE VIEW v_review_queue AS
        SELECT
            review_rank,
            filename_normalized,
            review_priority,
            review_score_percentile,
            review_reason,
            n_unique_participants,
            stomatal_index_iqr,
            uncertain_mark_fraction
        FROM image_review_priority
        WHERE review_priority IN ('red', 'yellow')
        ORDER BY review_rank;

        CREATE VIEW v_image_dashboard AS
        SELECT
            filename_normalized,
            review_priority,
            review_rank,
            review_reason,
            analysis_eligible,
            n_classifications_raw,
            n_unique_participants,
            n_additional_submissions_ignored,
            stomatal_index_median,
            stomatal_index_iqr,
            uncertain_mark_fraction,
            disagreement_level,
            uncertainty_level
        FROM image_review_priority;
        """
    )


def build_database(tables: dict[str, pd.DataFrame]) -> None:
    """Build a new database file and replace the previous generated version."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TEMP_DB_PATH.exists():
        TEMP_DB_PATH.unlink()

    # sqlite3.Connection's own context manager commits or rolls back but does
    # not close the connection. Explicit closing is required before Windows
    # will allow the temporary database file to be renamed.
    with closing(sqlite3.connect(TEMP_DB_PATH)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        for table_name, frame in tables.items():
            frame.to_sql(table_name, connection, index=False, if_exists="replace")

        metadata = pd.DataFrame([
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "classification_rows": len(tables["classifications_clean"]),
                "participant_image_rows": len(
                    tables["participant_image_contributions"]
                ),
                "image_rows": len(tables["image_review_priority"]),
                "threshold_rows": len(tables["quality_thresholds"]),
            }
        ])
        metadata.to_sql(
            "pipeline_metadata",
            connection,
            index=False,
            if_exists="replace",
        )

        create_indexes_and_views(connection)
        connection.commit()

    if DB_PATH.exists():
        DB_PATH.unlink()
    TEMP_DB_PATH.replace(DB_PATH)


def verify_database() -> None:
    """Run SQL checks against the completed database."""
    with closing(sqlite3.connect(DB_PATH)) as connection:
        objects = pd.read_sql_query(
            """
            SELECT type, name
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """,
            connection,
        )
        row_counts = {}
        for table_name in [
            "classifications_clean",
            "participant_image_contributions",
            "image_review_priority",
            "quality_thresholds",
        ]:
            count = connection.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()[0]
            row_counts[table_name] = count

        priority = pd.read_sql_query(
            """
            SELECT review_priority, image_count, percentage
            FROM v_priority_summary
            ORDER BY CASE review_priority
                WHEN 'green' THEN 1
                WHEN 'yellow' THEN 2
                WHEN 'red' THEN 3
            END
            """,
            connection,
        )

        top_five = pd.read_sql_query(
            """
            SELECT review_rank, filename_normalized,
                   review_priority, review_reason
            FROM v_review_queue
            ORDER BY review_rank
            LIMIT 5
            """,
            connection,
        )

    print("=== DATABASE OBJECTS ===")
    print(objects.to_string(index=False))

    print("\n=== ROW COUNTS ===")
    for table_name, count in row_counts.items():
        print(f"{table_name:32} {count}")

    print("\n=== SQL PRIORITY SUMMARY ===")
    print(priority.to_string(index=False))

    print("\n=== SQL TOP 5 REVIEW QUEUE ===")
    print(top_five.to_string(index=False))

    print(f"\nWrote: {DB_PATH.resolve()}")
    print(f"Database size: {DB_PATH.stat().st_size:,} bytes")


def main() -> None:
    tables = read_inputs()
    validate_inputs(tables)
    build_database(tables)
    verify_database()


if __name__ == "__main__":
    main()
