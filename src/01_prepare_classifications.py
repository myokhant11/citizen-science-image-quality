from pathlib import Path
import json
import re

import pandas as pd


IN_PATH = Path("data/raw/classification_export.csv")
OUT_PATH = Path("data/private/classifications_clean.csv")

EXPECTED_LABELS = {
    "Stoma": "count_stoma",
    "Regular cell": "count_regular_cell",
    "Not sure": "count_not_sure",
    "Unclear patch": "count_unclear_patch",
}


def normalize_filename(filename: str) -> str:
    """Return one consistent identifier for known filename variants."""
    base = re.sub(r"\.jpg$", "", str(filename), flags=re.I)
    base = re.sub(r"\s+copy$", "", base, flags=re.I)
    base = re.sub(r"\s*-\s*copy", "", base, flags=re.I)
    base = re.sub(r"_redo", "", base, flags=re.I)
    base = re.sub(r"gridrelocate", "grid", base, flags=re.I)
    base = re.sub(r"[\s_]+", "_", base).strip("_")
    return base.lower()


def parse_subject_data(subject_data_raw: str) -> dict:
    """Extract the subject-data filename without silently losing errors."""
    result = {
        "filename_original": "",
        "subject_data_subject_id": "",
        "subject_data_parse_ok": False,
        "has_multiple_subject_records": False,
    }

    try:
        subject_data = json.loads(subject_data_raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return result

    if not isinstance(subject_data, dict) or not subject_data:
        return result

    result["subject_data_parse_ok"] = True
    result["has_multiple_subject_records"] = len(subject_data) > 1

    subject_id, metadata = next(iter(subject_data.items()))
    result["subject_data_subject_id"] = str(subject_id)
    if isinstance(metadata, dict):
        result["filename_original"] = str(metadata.get("Filename", "") or "")

    return result


def parse_annotations(annotations_raw: str) -> dict:
    """Extract annotation counts and retain structural quality information."""
    result = {
        column: 0 for column in EXPECTED_LABELS.values()
    }
    result.update({
        "annotation_parse_ok": False,
        "task_sequence": "",
        "n_annotation_tasks": 0,
        "t1_answer": "",
        "t4_present": False,
        "has_duplicate_task": False,
        "unexpected_labels": "",
    })

    try:
        annotations = json.loads(annotations_raw or "[]")
    except (json.JSONDecodeError, TypeError):
        return result

    if not isinstance(annotations, list):
        return result

    result["annotation_parse_ok"] = True
    task_ids = []
    t1_answers = []
    unexpected_labels = []

    for task in annotations:
        if not isinstance(task, dict):
            continue

        task_id = str(task.get("task", "") or "")
        task_ids.append(task_id)
        value = task.get("value")

        if task_id == "T1" and value is not None and not isinstance(value, (list, dict)):
            t1_answers.append(str(value))
        if task_id == "T4":
            result["t4_present"] = True

        if not isinstance(value, list):
            continue

        for mark in value:
            if not isinstance(mark, dict):
                continue
            label = mark.get("tool_label") or mark.get("label") or "unknown"
            output_column = EXPECTED_LABELS.get(label)
            if output_column is None:
                unexpected_labels.append(str(label))
            else:
                result[output_column] += 1

    result["task_sequence"] = ",".join(task_ids)
    result["n_annotation_tasks"] = len(task_ids)
    result["t1_answer"] = " | ".join(t1_answers)
    result["has_duplicate_task"] = len(task_ids) != len(set(task_ids))
    result["unexpected_labels"] = " | ".join(sorted(set(unexpected_labels)))
    return result


def participant_source_key(row: pd.Series) -> str:
    """Create an internal grouping key without exposing it in the output."""
    user_id = str(row.get("user_id", "") or "")
    user_name = str(row.get("user_name", "") or "")

    if user_id.isdigit():
        return f"registered:{user_id}"
    if user_name:
        return f"anonymous:{user_name}"
    return f"unknown:{row['classification_id']}"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(
        IN_PATH,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    subject_fields = pd.DataFrame(
        raw["subject_data"].apply(parse_subject_data).tolist()
    )
    annotation_fields = pd.DataFrame(
        raw["annotations"].apply(parse_annotations).tolist()
    )

    source_keys = raw.apply(participant_source_key, axis=1)
    participant_codes, _ = pd.factorize(source_keys, sort=True)

    output = pd.DataFrame({
        "classification_id": raw["classification_id"],
        "participant_id": [f"P{code + 1:04d}" for code in participant_codes],
        "workflow_id": raw["workflow_id"],
        "workflow_version": raw["workflow_version"],
        "created_at": raw["created_at"],
        "subject_id": raw["subject_ids"],
    })

    output = pd.concat([output, subject_fields, annotation_fields], axis=1)
    output["filename_normalized"] = output["filename_original"].apply(normalize_filename)
    output["subject_id_matches_subject_data"] = (
        output["subject_id"] == output["subject_data_subject_id"]
    )

    count_columns = list(EXPECTED_LABELS.values())
    output["total_marks"] = output[count_columns].sum(axis=1)
    output["is_empty"] = output["total_marks"] == 0

    participant_image = output.groupby(
        ["participant_id", "filename_normalized"], sort=False
    )
    output["participant_image_submission_number"] = participant_image.cumcount() + 1
    output["n_submissions_by_participant_image"] = (
        participant_image["classification_id"].transform("size")
    )
    output["is_repeat_group"] = output["n_submissions_by_participant_image"] > 1
    output["is_additional_submission"] = (
        output["participant_image_submission_number"] > 1
    )

    preferred_order = [
        "classification_id",
        "participant_id",
        "workflow_id",
        "workflow_version",
        "created_at",
        "subject_id",
        "subject_data_subject_id",
        "filename_original",
        "filename_normalized",
        *count_columns,
        "t1_answer",
        "t4_present",
        "task_sequence",
        "n_annotation_tasks",
        "annotation_parse_ok",
        "subject_data_parse_ok",
        "subject_id_matches_subject_data",
        "has_multiple_subject_records",
        "has_duplicate_task",
        "unexpected_labels",
        "total_marks",
        "is_empty",
        "participant_image_submission_number",
        "n_submissions_by_participant_image",
        "is_repeat_group",
        "is_additional_submission",
    ]
    output = output[preferred_order]

    assert len(output) == len(raw), "Rows were lost during extraction."
    assert output["classification_id"].is_unique, "Classification IDs are not unique."
    assert not output[["classification_id", "participant_id"]].isna().any().any()

    output.to_csv(OUT_PATH, index=False)

    print(f"Input rows:                {len(raw)}")
    print(f"Output rows:               {len(output)}")
    print(f"Unique participants:       {output['participant_id'].nunique()}")
    print(f"Original subject IDs:      {output['subject_id'].nunique()}")
    print(f"Original filenames:        {output['filename_original'].nunique()}")
    print(f"Normalized images:         {output['filename_normalized'].nunique()}")
    print(f"Annotation parse failures: {(~output['annotation_parse_ok']).sum()}")
    print(f"Subject parse failures:    {(~output['subject_data_parse_ok']).sum()}")
    print(f"Empty classifications:     {output['is_empty'].sum()}")
    print(f"Duplicate-task rows:       {output['has_duplicate_task'].sum()}")
    print(f"Rows in repeat groups:     {output['is_repeat_group'].sum()}")
    print(f"Additional submissions:    {output['is_additional_submission'].sum()}")
    print(f"Wrote: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
