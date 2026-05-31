from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from portfolio_schema import (
    BUCKET_COLUMNS,
    CATEGORICAL_COLUMNS,
    build_allocation_dataset,
)
from run_end_to_end import load_end_to_end_resources, run_end_to_end

REPO_ROOT = Path(__file__).resolve().parents[1]

def _build_summary_rows(
    df: pd.DataFrame,
    predicted_allocations: np.ndarray,
    model_risk: np.ndarray,
    allocation_risk: np.ndarray,
    used_risk: np.ndarray,
    predicted_risky_share: np.ndarray,
    aux_allocations: np.ndarray,
    aux_risk: np.ndarray,
    quality: pd.DataFrame,
    recommendations: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    alloc_mae = np.mean(np.abs(predicted_allocations - aux_allocations), axis=1)

    for idx, recommendation in enumerate(recommendations):
        top_ranked = recommendation["ranked_products"][0] if recommendation["ranked_products"] else {}
        top_basket = recommendation["optimized_basket"][0] if recommendation["optimized_basket"] else {}
        row = {
            "CASEID": int(df.iloc[idx]["CASEID"]) if "CASEID" in df.columns else idx,
            "predicted_risk_level_model": int(model_risk[idx]),
            "predicted_risk_level_allocation": int(allocation_risk[idx]),
            "risk_level_used_for_recommendation": int(used_risk[idx]),
            "predicted_risky_share": float(predicted_risky_share[idx]),
            "aux_risk_label": int(aux_risk[idx]),
            "aux_allocation_mae": float(alloc_mae[idx]),
            "rescaled_overlap_flag": bool(quality.iloc[idx]["rescaled_overlap_flag"]),
            "coverage_before_rescale": float(quality.iloc[idx]["coverage_before_rescale"]),
            "coverage_after_rescale": float(quality.iloc[idx]["coverage_after_rescale"]),
            "top_ranked_product_id": top_ranked.get("product_id"),
            "top_ranked_product_name": top_ranked.get("name"),
            "top_basket_product_id": top_basket.get("product_id"),
            "top_basket_weight": top_basket.get("weight"),
        }
        for bucket_idx, bucket in enumerate(BUCKET_COLUMNS):
            row[f"pred_{bucket}"] = float(predicted_allocations[idx, bucket_idx])
            row[f"aux_{bucket}"] = float(aux_allocations[idx, bucket_idx])
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=REPO_ROOT / "dataset" / "test.csv",
        help="Use dataset/test.csv for held-out evaluation, or anchor_labeled_data.csv for full-dataset batch inference.",
    )
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "dataset" / "processed")
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--checkpoint-prefix", type=str, default=None)
    parser.add_argument("--naverpay-path", type=Path, default=None)
    parser.add_argument("--pykrx-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--allow-cma", action="store_true")
    parser.add_argument("--risk-source", choices=["model", "allocation"], default="model")
    parser.add_argument("--anchor-csv", type=Path, default=REPO_ROOT / "dataset" / "train.csv")
    parser.add_argument("--disable-knn-smoothing", action="store_true")
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--knn-alpha", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-csv", type=Path, default=REPO_ROOT / "checkpoints" / "batch_end_to_end_summary.csv")
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / "checkpoints" / "batch_end_to_end_details.json")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    build_result = build_allocation_dataset(df)
    x_cat = torch.tensor(build_result.categorical_frame.values, dtype=torch.long)
    aux_allocations = build_result.allocation_frame.values.astype(np.float32)
    aux_risk = build_result.labels.values.astype(np.int64)

    resources = load_end_to_end_resources(
        processed_dir=args.processed_dir,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_prefix=args.checkpoint_prefix,
        naverpay_path=args.naverpay_path,
        pykrx_path=args.pykrx_path,
        device=args.device,
        anchor_csv=args.anchor_csv,
        load_knn_anchors=not args.disable_knn_smoothing,
        current_x_cat_tensor=x_cat,
    )

    results = []
    for row in build_result.categorical_frame.to_dict(orient="records"):
        profile = {column: int(row[column]) for column in CATEGORICAL_COLUMNS}
        profile["options"] = {
            "top_k": args.top_k,
            "allow_cma": args.allow_cma,
        }
        results.append(
            run_end_to_end(
                profile,
                resources=resources,
                risk_source=args.risk_source,
                disable_knn_smoothing=args.disable_knn_smoothing,
                knn_k=args.knn_k,
                knn_alpha=args.knn_alpha,
            )
        )

    predicted_allocations_np = np.array(
        [
            [result["predicted_allocation"][bucket] for bucket in BUCKET_COLUMNS]
            for result in results
        ],
        dtype=np.float32,
    )
    predicted_risky_share_np = np.array(
        [result["predicted_risky_share"] for result in results],
        dtype=np.float32,
    )
    model_risk_np = np.array(
        [result["predicted_risk_level_model"] for result in results],
        dtype=np.int64,
    )
    allocation_risk_np = np.array(
        [result["predicted_risk_level_allocation"] for result in results],
        dtype=np.int64,
    )
    risk_for_recs_np = np.array(
        [result["risk_level_used_for_recommendation"] for result in results],
        dtype=np.int64,
    )
    recommendations = [result["recommendation"] for result in results]

    summary_rows = _build_summary_rows(
        df=df,
        predicted_allocations=predicted_allocations_np,
        model_risk=model_risk_np,
        allocation_risk=allocation_risk_np,
        used_risk=risk_for_recs_np,
        predicted_risky_share=predicted_risky_share_np,
        aux_allocations=aux_allocations,
        aux_risk=aux_risk,
        quality=build_result.quality_frame,
        recommendations=recommendations,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(summary_rows).to_csv(args.output_csv, index=False)

    details = {
        "input_csv": str(args.input_csv),
        "checkpoint_prefix": resources.checkpoint_prefix,
        "num_examples": int(len(df)),
        "validation_warnings": resources.validation["warnings"],
        "model_config": resources.checkpoint_meta["model_config"],
        "summary_metrics": {
            "pred_vs_aux_allocation_mae_mean": float(
                np.mean(np.abs(predicted_allocations_np - aux_allocations))
            ),
            "pred_vs_aux_model_risk_match_rate": float(np.mean(model_risk_np == aux_risk)),
            "pred_vs_aux_allocation_risk_match_rate": float(np.mean(allocation_risk_np == aux_risk)),
            "overlap_rescaled_share": float(build_result.quality_frame["rescaled_overlap_flag"].mean()),
        },
        "risk_source_used_for_recommendation": args.risk_source,
        "knn_smoothing": {
            "enabled": not args.disable_knn_smoothing,
            "anchor_csv": str(args.anchor_csv),
            "k": args.knn_k,
            "alpha": args.knn_alpha,
        },
        "rows": summary_rows,
    }
    args.output_json.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved summary csv to {args.output_csv}")
    print(f"saved detail json to {args.output_json}")


if __name__ == "__main__":
    main()
