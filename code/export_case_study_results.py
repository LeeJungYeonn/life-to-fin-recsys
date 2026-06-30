from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from portfolio_schema import BUCKET_COLUMNS, CATEGORICAL_COLUMNS, build_allocation_dataset
from run_end_to_end import load_end_to_end_resources, run_end_to_end


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_IDS = [165546, 176876, 190931, 236386, 222331]
RISK_LABELS_ZERO_BASED = [0, 1, 2, 3, 4]


def _format_pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _row_to_profile(row: pd.Series, *, top_k: int, allow_cma: bool) -> dict:
    return {
        "categorical_features": {
            column: int(pd.to_numeric(row[column], errors="coerce"))
            for column in CATEGORICAL_COLUMNS
        },
        "options": {
            "top_k": top_k,
            "allow_cma": allow_cma,
        },
    }


def _basket_summary(basket: list[dict[str, object]]) -> str:
    parts = []
    for product in basket:
        parts.append(
            "{name} ({bucket}, {weight})".format(
                name=product.get("name"),
                bucket=product.get("bucket"),
                weight=_format_pct(float(product.get("weight", 0.0))),
            )
        )
    return "; ".join(parts)


def _select_case_ids_from_summary(
    summary_path: Path,
    *,
    risk_column: str,
    allow_missing: bool,
) -> tuple[list[int], list[int]]:
    summary = pd.read_csv(summary_path)
    missing_columns = [column for column in ["CASEID", risk_column, "aux_allocation_mae"] if column not in summary.columns]
    if missing_columns:
        raise ValueError(f"Missing column(s) in {summary_path}: {missing_columns}")

    selected = []
    missing_labels = []
    for label in RISK_LABELS_ZERO_BASED:
        candidates = summary[summary[risk_column].astype(int) == label]
        if candidates.empty:
            missing_labels.append(label + 1)
            continue
        best = candidates.sort_values("aux_allocation_mae").iloc[0]
        selected.append(int(best["CASEID"]))

    if missing_labels and not allow_missing:
        raise ValueError(
            "No case(s) found for displayed predicted risk label(s) "
            f"{missing_labels} in {summary_path}. "
            "Use --allow-missing-risk-labels to export available labels only."
        )
    return selected, missing_labels


def _markdown_case(case: dict[str, object]) -> str:
    input_items = case["input_features"]
    predicted_allocation = case["predicted_allocation"]
    true_allocation = case["true_allocation"]
    basket = case["recommended_products"]

    lines = [
        f"## CASEID {case['CASEID']} - Predicted Risk Label {case['predicted_risk_label_1_to_5']}",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| true risk label | {case['true_risk_label_1_to_5']} |",
        f"| predicted risk label | {case['predicted_risk_label_1_to_5']} |",
        f"| risky share | {_format_pct(case['predicted_risky_share'])} |",
        "",
        "### Input: non-financial categorical features",
        "",
        "| Feature | Value |",
        "| --- | ---: |",
    ]
    for column, value in input_items.items():
        lines.append(f"| {column} | {value} |")

    lines.extend(
        [
            "",
            "### Allocation",
            "",
            "| Bucket | True | Predicted |",
            "| --- | ---: | ---: |",
        ]
    )
    for bucket in BUCKET_COLUMNS:
        lines.append(
            f"| {bucket} | {_format_pct(true_allocation[bucket])} | "
            f"{_format_pct(predicted_allocation[bucket])} |"
        )

    lines.extend(
        [
            "",
            "### Recommended Products",
            "",
            "| Rank | Product | Bucket | Weight | Category | Type | Score |",
            "| ---: | --- | --- | ---: | --- | --- | ---: |",
        ]
    )
    for rank, product in enumerate(basket, start=1):
        lines.append(
            "| {rank} | {name} | {bucket} | {weight} | {category} | {ptype} | {score:.4f} |".format(
                rank=rank,
                name=product.get("name"),
                bucket=product.get("bucket"),
                weight=_format_pct(float(product.get("weight", 0.0))),
                category=product.get("category"),
                ptype=product.get("product_type"),
                score=float(product.get("score", 0.0)),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=REPO_ROOT / "dataset" / "splits" / "test.csv")
    parser.add_argument("--case-ids", type=int, nargs="+", default=None)
    parser.add_argument(
        "--sample-by",
        choices=["predicted-risk", "true-risk", "case-ids"],
        default="predicted-risk",
        help="Default uses checkpoints/batch_end_to_end_summary.csv to select one CASEID per predicted risk label.",
    )
    parser.add_argument(
        "--selection-summary",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "batch_end_to_end_summary.csv",
    )
    parser.add_argument("--allow-missing-risk-labels", action="store_true")
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "dataset" / "processed")
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--checkpoint-prefix", type=str, default="allocation_best")
    parser.add_argument("--naverpay-path", type=Path, default=None)
    parser.add_argument("--pykrx-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--risk-source", choices=["model", "allocation"], default="model")
    parser.add_argument("--anchor-csv", type=Path, default=REPO_ROOT / "dataset" / "splits" / "train.csv")
    parser.add_argument("--enable-knn-smoothing", action="store_true")
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--knn-alpha", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--allow-cma", action="store_true")
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "case_study_results.md",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "case_study_results.csv",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.case_ids is not None:
        case_ids = args.case_ids
        missing_display_labels = []
    elif args.sample_by == "predicted-risk":
        case_ids, missing_display_labels = _select_case_ids_from_summary(
            args.selection_summary,
            risk_column="predicted_risk_level_model",
            allow_missing=args.allow_missing_risk_labels,
        )
    elif args.sample_by == "true-risk":
        case_ids, missing_display_labels = _select_case_ids_from_summary(
            args.selection_summary,
            risk_column="aux_risk_label",
            allow_missing=args.allow_missing_risk_labels,
        )
    else:
        case_ids = DEFAULT_CASE_IDS
        missing_display_labels = []

    missing = [case_id for case_id in case_ids if case_id not in set(df["CASEID"].astype(int))]
    if missing:
        raise ValueError(f"CASEID(s) not found in {args.input_csv}: {missing}")

    selected = (
        df[df["CASEID"].astype(int).isin(case_ids)]
        .copy()
        .set_index("CASEID")
        .loc[case_ids]
        .reset_index()
    )
    build_result = build_allocation_dataset(selected)

    resources = load_end_to_end_resources(
        processed_dir=args.processed_dir,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_prefix=args.checkpoint_prefix,
        naverpay_path=args.naverpay_path,
        pykrx_path=args.pykrx_path,
        device=args.device,
        anchor_csv=args.anchor_csv,
        load_knn_anchors=args.enable_knn_smoothing,
    )

    cases = []
    csv_rows = []
    for idx, row in selected.iterrows():
        case_id = int(row["CASEID"])
        profile = _row_to_profile(row, top_k=args.top_k, allow_cma=args.allow_cma)
        result = run_end_to_end(
            profile,
            resources=resources,
            risk_source=args.risk_source,
            anchor_csv=args.anchor_csv,
            disable_knn_smoothing=not args.enable_knn_smoothing,
            knn_k=args.knn_k,
            knn_alpha=args.knn_alpha,
        )

        true_allocation = {
            bucket: float(build_result.allocation_frame.iloc[idx][bucket])
            for bucket in BUCKET_COLUMNS
        }
        true_risk_zero_based = int(build_result.labels.iloc[idx])
        predicted_risk_zero_based = int(result["risk_level_used_for_recommendation"])
        predicted_allocation = {
            bucket: float(result["predicted_allocation"][bucket])
            for bucket in BUCKET_COLUMNS
        }
        basket = result["recommendation"]["optimized_basket"]
        case = {
            "CASEID": case_id,
            "input_features": profile["categorical_features"],
            "true_risk_label_0_to_4": true_risk_zero_based,
            "true_risk_label_1_to_5": true_risk_zero_based + 1,
            "predicted_risk_label_0_to_4": predicted_risk_zero_based,
            "predicted_risk_label_1_to_5": predicted_risk_zero_based + 1,
            "predicted_risky_share": float(result["predicted_risky_share"]),
            "true_allocation": true_allocation,
            "predicted_allocation": predicted_allocation,
            "recommended_products": basket,
        }
        cases.append(case)

        csv_row = {
            "CASEID": case_id,
            "true_risk_label_1_to_5": true_risk_zero_based + 1,
            "predicted_risk_label_1_to_5": predicted_risk_zero_based + 1,
            "predicted_risky_share": float(result["predicted_risky_share"]),
            "recommended_products": _basket_summary(basket),
        }
        for column, value in profile["categorical_features"].items():
            csv_row[f"input_{column}"] = value
        for bucket in BUCKET_COLUMNS:
            csv_row[f"true_{bucket}"] = true_allocation[bucket]
            csv_row[f"pred_{bucket}"] = predicted_allocation[bucket]
        csv_rows.append(csv_row)

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "# Case Study Results",
        "",
        f"- input_csv: `{args.input_csv}`",
        f"- checkpoint_prefix: `{resources.checkpoint_prefix}`",
        f"- sample_by: `{args.sample_by}`",
        f"- selected_case_ids: `{case_ids}`",
        f"- risk_source: `{args.risk_source}`",
        f"- knn_smoothing: `{args.enable_knn_smoothing}`",
        f"- top_k: `{args.top_k}`",
        "",
    ]
    if missing_display_labels:
        header.extend(
            [
                "## Missing Predicted Risk Labels",
                "",
                "No test cases were found for displayed risk label(s): "
                + ", ".join(str(label) for label in missing_display_labels),
                "",
            ]
        )
    args.output_md.write_text(
        "\n".join(header + [_markdown_case(case) for case in cases]),
        encoding="utf-8-sig",
    )
    pd.DataFrame(csv_rows).to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print(f"saved markdown to {args.output_md}")
    print(f"saved csv to {args.output_csv}")


if __name__ == "__main__":
    main()
