from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from checkpoint_utils import load_dual_encoder_checkpoint
from contrastive_utils import ordinal_logits_to_label
from portfolio_schema import BUCKET_COLUMNS, CATEGORICAL_COLUMNS, build_allocation_dataset
from recsys.naverpay_catalog import load_default_snapshot, load_snapshot as load_naver_snapshot
from recsys.pykrx_catalog import find_default_snapshot_path, load_snapshot as load_pykrx_snapshot
from recsys.ranker import UserRequest, recommend_products
from run_end_to_end import _heuristic_risk_from_allocation, _resolve_checkpoint_prefix

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_products(naverpay_path: Path | None, pykrx_path: Path | None):
    products = []
    if naverpay_path is not None:
        products.extend(load_naver_snapshot(naverpay_path))
    else:
        products.extend(load_default_snapshot())

    resolved_pykrx_path = pykrx_path
    if resolved_pykrx_path is None:
        resolved_pykrx_path = find_default_snapshot_path()
    if resolved_pykrx_path is not None and resolved_pykrx_path.exists():
        products.extend(load_pykrx_snapshot(resolved_pykrx_path))
    return products


def _build_summary_rows(
    df: pd.DataFrame,
    predicted_allocations: np.ndarray,
    direct_risk: np.ndarray,
    heuristic_risk: np.ndarray,
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
            "predicted_risk_level_direct": int(direct_risk[idx]),
            "predicted_risk_level_heuristic": int(heuristic_risk[idx]),
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

    checkpoint_prefix = _resolve_checkpoint_prefix(args.checkpoint_dir, args.checkpoint_prefix)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    cardinalities = torch.load(args.processed_dir / "train_cardinalities.pt").tolist()

    source_encoder, _, checkpoint_meta, validation = load_dual_encoder_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        current_categorical_cols=CATEGORICAL_COLUMNS,
        current_categorical_cardinalities=cardinalities,
        current_x_cat_tensor=x_cat,
        current_split=None,
        strict=False,
        prefix=checkpoint_prefix,
    )

    predicted_allocations = []
    direct_risk = []
    with torch.no_grad():
        for start in range(0, len(x_cat), args.batch_size):
            batch_x_cat = x_cat[start : start + args.batch_size].to(device)
            source_output = source_encoder(batch_x_cat)
            predicted_allocations.append(source_output.allocation_probs.cpu().numpy())
            direct_risk.append(ordinal_logits_to_label(source_output.risk_logits).cpu().numpy())

    predicted_allocations_np = np.concatenate(predicted_allocations, axis=0)
    direct_risk_np = np.concatenate(direct_risk, axis=0)
    heuristic_risk_np = np.array(
        [_heuristic_risk_from_allocation(allocation) for allocation in predicted_allocations_np],
        dtype=np.int64,
    )

    products = _load_products(args.naverpay_path, args.pykrx_path)
    recommendations = []
    for idx, allocation in enumerate(predicted_allocations_np):
        recommendation = recommend_products(
            products,
            UserRequest(
                risk_level=int(direct_risk_np[idx]),
                predicted_allocation=allocation,
                allow_cma=args.allow_cma,
            ),
            top_k=args.top_k,
        )
        recommendations.append(recommendation)

    summary_rows = _build_summary_rows(
        df=df,
        predicted_allocations=predicted_allocations_np,
        direct_risk=direct_risk_np,
        heuristic_risk=heuristic_risk_np,
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
        "checkpoint_prefix": checkpoint_prefix,
        "num_examples": int(len(df)),
        "validation_warnings": validation["warnings"],
        "model_config": checkpoint_meta["model_config"],
        "summary_metrics": {
            "pred_vs_aux_allocation_mae_mean": float(
                np.mean(np.abs(predicted_allocations_np - aux_allocations))
            ),
            "pred_vs_aux_risk_match_rate": float(np.mean(direct_risk_np == aux_risk)),
            "overlap_rescaled_share": float(build_result.quality_frame["rescaled_overlap_flag"].mean()),
        },
        "rows": summary_rows,
    }
    args.output_json.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved summary csv to {args.output_csv}")
    print(f"saved detail json to {args.output_json}")


if __name__ == "__main__":
    main()
