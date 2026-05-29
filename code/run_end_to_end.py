from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from checkpoint_utils import load_dual_encoder_checkpoint
from contrastive_utils import ordinal_logits_to_label
from portfolio_schema import (
    BUCKET_COLUMNS,
    CATEGORICAL_COLUMNS,
    derive_risk_label_from_allocation_vector,
)
from recsys.naverpay_catalog import load_default_snapshot, load_snapshot as load_naver_snapshot
from recsys.pykrx_catalog import find_default_snapshot_path, load_snapshot as load_pykrx_snapshot
from recsys.ranker import UserRequest, recommend_products

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_checkpoint_prefix(checkpoint_dir: Path, requested_prefix: str | None) -> str:
    if requested_prefix:
        return requested_prefix

    for candidate in ["allocation_best", "smoke_allocation", "best"]:
        meta_path = checkpoint_dir / f"{candidate}_checkpoint_meta.json"
        if meta_path.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint meta found in {checkpoint_dir}")


def _load_profile(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_input_tensor(profile: dict, cardinalities: list[int]) -> torch.Tensor:
    feature_values = profile.get("categorical_features", profile)
    missing = [column for column in CATEGORICAL_COLUMNS if column not in feature_values]
    if missing:
        raise KeyError(f"Missing categorical feature(s): {missing}")

    values = []
    for index, column in enumerate(CATEGORICAL_COLUMNS):
        value = int(feature_values[column])
        if value < 0:
            raise ValueError(f"{column} must be >= 0, got {value}")
        if value >= cardinalities[index]:
            raise ValueError(
                f"{column}={value} is outside trained range [0, {cardinalities[index] - 1}]"
            )
        values.append(value)
    return torch.tensor([values], dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Path to user profile JSON")
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "dataset" / "processed")
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--checkpoint-prefix", type=str, default=None)
    parser.add_argument("--naverpay-path", type=Path, default=None)
    parser.add_argument("--pykrx-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--risk-source", choices=["heuristic", "direct"], default="heuristic")
    args = parser.parse_args()

    profile = _load_profile(args.input)
    checkpoint_prefix = _resolve_checkpoint_prefix(args.checkpoint_dir, args.checkpoint_prefix)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cardinalities = torch.load(args.processed_dir / "train_cardinalities.pt").tolist()
    x_cat = _build_input_tensor(profile, cardinalities)

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

    with torch.no_grad():
        source_output = source_encoder(x_cat.to(device))

    allocation = source_output.allocation_probs.squeeze(0).cpu().numpy()
    direct_risk_level = int(ordinal_logits_to_label(source_output.risk_logits).item())
    heuristic_risk_level = derive_risk_label_from_allocation_vector(allocation)

    options = profile.get("options", {})
    top_k = int(options.get("top_k", 5))
    allow_cma = bool(options.get("allow_cma", False))
    default_risk_level = heuristic_risk_level if args.risk_source == "heuristic" else direct_risk_level
    risk_level_for_recs = int(options.get("risk_level_override", default_risk_level))

    products = []
    if args.naverpay_path is not None:
        products.extend(load_naver_snapshot(args.naverpay_path))
    else:
        products.extend(load_default_snapshot())

    pykrx_path = args.pykrx_path
    if pykrx_path is None:
        pykrx_path = find_default_snapshot_path()
    if pykrx_path is not None and pykrx_path.exists():
        products.extend(load_pykrx_snapshot(pykrx_path))

    recommendation = recommend_products(
        products,
        UserRequest(
            risk_level=risk_level_for_recs,
            predicted_allocation=allocation,
            allow_cma=allow_cma,
        ),
        top_k=top_k,
    )

    result = {
        "input_path": str(args.input),
        "checkpoint_prefix": checkpoint_prefix,
        "validation_warnings": validation["warnings"],
        "categorical_features": {
            column: int(value)
            for column, value in zip(CATEGORICAL_COLUMNS, x_cat.squeeze(0).tolist())
        },
        "predicted_allocation": {
            bucket: float(value) for bucket, value in zip(BUCKET_COLUMNS, allocation.tolist())
        },
        "predicted_risk_level_direct": direct_risk_level,
        "predicted_risk_level_heuristic": heuristic_risk_level,
        "risk_level_used_for_recommendation": risk_level_for_recs,
        "risk_source_used_for_recommendation": args.risk_source,
        "options": {
            "top_k": top_k,
            "allow_cma": allow_cma,
        },
        "recommendation": recommendation,
        "model_config": checkpoint_meta["model_config"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
