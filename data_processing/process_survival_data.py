import argparse
import os

import pandas as pd

try:
    from .process_clinical_data_qu import normalize_cancer_type
except ImportError:  # Allow direct execution: python data_processing/process_survival_data.py
    from process_clinical_data_qu import normalize_cancer_type


DEFAULT_SAMPLE_COL = "样本编号"
DEFAULT_TIME_COL = "随访时间(月)"
DEFAULT_EVENT_COL = "随访生存状态\n（存活/死亡）"
DEFAULT_GROUP_COL = "分组"


def resolve_column(df: pd.DataFrame, requested: str, table_name: str) -> str:
    """Resolve a column exactly or by ignoring whitespace/newlines in its header."""
    if requested in df.columns:
        return requested

    normalized_requested = "".join(str(requested).split())
    matches = [
        column
        for column in df.columns
        if "".join(str(column).split()) == normalized_requested
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"{table_name} has multiple columns matching {requested!r} after whitespace normalization: {matches}"
        )
    raise ValueError(
        f"{table_name} is missing required column: {requested}. "
        f"Available columns: {df.columns.tolist()}"
    )


def normalize_survival_event(value):
    if pd.isna(value):
        return pd.NA

    if isinstance(value, (int, float)) and not pd.isna(value):
        if int(value) in {0, 1}:
            return int(value)

    text = str(value).strip().lower()
    if text in {"0", "存活", "alive", "living", "censored", "no", "false"}:
        return 0
    if text in {"1", "死亡", "dead", "deceased", "event", "yes", "true"}:
        return 1
    return pd.NA


def build_survival_dataset(
    clinical_df: pd.DataFrame,
    pathology_df: pd.DataFrame,
    sample_col: str = DEFAULT_SAMPLE_COL,
    time_col: str = DEFAULT_TIME_COL,
    event_col: str = DEFAULT_EVENT_COL,
    group_col: str = DEFAULT_GROUP_COL,
    keep_missing_survival_time: bool = False,
    drop_missing_survival_event: bool = True,
) -> pd.DataFrame:
    clinical_sample_col = resolve_column(clinical_df, sample_col, "Clinical file")
    clinical_time_col = resolve_column(clinical_df, time_col, "Clinical file")
    clinical_event_col = resolve_column(clinical_df, event_col, "Clinical file")
    pathology_sample_col = resolve_column(pathology_df, sample_col, "Pathology file")
    pathology_group_col = resolve_column(pathology_df, group_col, "Pathology file")

    clinical = clinical_df[[clinical_sample_col, clinical_time_col, clinical_event_col]].copy()
    clinical = clinical.rename(
        columns={
            clinical_sample_col: "_sample_id",
            clinical_time_col: "_survival_time",
            clinical_event_col: "_survival_event",
        }
    )
    clinical = clinical[clinical["_sample_id"].notna()].copy()
    clinical["_sample_id"] = clinical["_sample_id"].astype(str).str.strip()
    clinical = clinical[clinical["_sample_id"] != ""].drop_duplicates(subset=["_sample_id"], keep="first")

    pathology = pathology_df[[pathology_sample_col, pathology_group_col]].copy()
    pathology = pathology.rename(
        columns={
            pathology_sample_col: "_sample_id",
            pathology_group_col: "_group",
        }
    )
    pathology = pathology[pathology["_sample_id"].notna()].copy()
    pathology["_sample_id"] = pathology["_sample_id"].astype(str).str.strip()
    pathology = pathology[pathology["_sample_id"] != ""].drop_duplicates(subset=["_sample_id"], keep="first")

    merged = clinical.merge(pathology, how="left", on="_sample_id")
    result = pd.DataFrame(
        {
            "sample_id": merged["_sample_id"],
            "cancer_type": merged["_group"].map(normalize_cancer_type),
            "survival_time": pd.to_numeric(merged["_survival_time"], errors="coerce"),
            "survival_event": merged["_survival_event"].map(normalize_survival_event),
        }
    )

    if not keep_missing_survival_time:
        result = result.dropna(subset=["survival_time"]).copy()
    if drop_missing_survival_event:
        result = result.dropna(subset=["survival_event"]).copy()

    return result.reset_index(drop=True)


def read_table(path: str, sep: str | None = None) -> pd.DataFrame:
    if sep is None:
        sep = "\t" if path.lower().endswith((".tsv", ".tsv.gz")) else ","
    return pd.read_csv(path, sep=sep)


def parse_args():
    parser = argparse.ArgumentParser(description="Build the standard OncoWM survival label table from cohort data.")
    parser.add_argument("--clinical", required=True, help="Path to the cohort follow-up CSV/TSV")
    parser.add_argument("--pathology", required=True, help="Path to the cohort pathology/group CSV/TSV")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--clinical-sep", default=None, help="Clinical file separator. Default: infer from extension")
    parser.add_argument("--pathology-sep", default=None, help="Pathology file separator. Default: infer from extension")
    parser.add_argument("--sample-col", default=DEFAULT_SAMPLE_COL, help=f"Sample ID column shared by both files. Default: {DEFAULT_SAMPLE_COL}")
    parser.add_argument("--time-col", default=DEFAULT_TIME_COL, help=f"Survival time column. Default: {DEFAULT_TIME_COL}")
    parser.add_argument("--event-col", default=DEFAULT_EVENT_COL, help="Survival event/status column")
    parser.add_argument("--group-col", default=DEFAULT_GROUP_COL, help=f"Cancer group column in pathology file. Default: {DEFAULT_GROUP_COL}")
    parser.add_argument("--keep-missing-survival-time", action="store_true", help="Keep rows whose survival_time is missing or non-numeric")
    parser.add_argument("--keep-missing-survival-event", action="store_true", help="Keep unrecognized/missing events (not valid for final model input)")
    return parser.parse_args()


def main():
    args = parse_args()
    clinical_df = read_table(args.clinical, args.clinical_sep)
    pathology_df = read_table(args.pathology, args.pathology_sep)
    result = build_survival_dataset(
        clinical_df,
        pathology_df,
        sample_col=args.sample_col,
        time_col=args.time_col,
        event_col=args.event_col,
        group_col=args.group_col,
        keep_missing_survival_time=args.keep_missing_survival_time,
        drop_missing_survival_event=not args.keep_missing_survival_event,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} rows to: {args.output}")
    print(result["cancer_type"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
