from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from checkpoint_utils import load_dual_encoder_checkpoint
from portfolio_schema import (
    BUCKET_COLUMNS,
    CATEGORICAL_COLUMNS,
    derive_risk_label_from_allocation_vector,
    risky_share_to_bucket,
)
from profile_knn import load_anchor_arrays, smooth_with_profile_knn
from recsys.naverpay_catalog import load_default_snapshot, load_snapshot as load_naver_snapshot
from recsys.pykrx_catalog import find_default_snapshot_path, load_snapshot as load_pykrx_snapshot
from recsys.ranker import UserRequest, recommend_products

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class EndToEndResources:
    source_encoder: torch.nn.Module
    checkpoint_meta: dict
    validation: dict
    cardinalities: list[int]
    products: list
    device: torch.device
    checkpoint_prefix: str
    x_anchor_cat: np.ndarray | None = None
    anchor_allocs: np.ndarray | None = None
    anchor_risky_shares: np.ndarray | None = None
    anchor_csv: Path | None = None


def _resolve_checkpoint_prefix(checkpoint_dir: Path, requested_prefix: str | None) -> str:
    if requested_prefix:
        return requested_prefix

    candidate = "allocation_best"
    meta_path = checkpoint_dir / f"{candidate}_checkpoint_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_config = meta.get("model_config", {})
        if (
            int(model_config.get("target_input_dim", 0)) == len(BUCKET_COLUMNS)
            and model_config.get("risk_head") == "risky_share_regression"
        ):
            return candidate
    raise FileNotFoundError(
        f"No compatible allocation_best checkpoint meta found in {checkpoint_dir}. "
        "Run code/train_allocation.py first."
    )


def _load_profile(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_input_tensor(profile: dict, cardinalities: list[int], *, coerce: bool = False) -> torch.Tensor:
    feature_values = profile.get("categorical_features", profile)
    if not coerce:
        missing = [column for column in CATEGORICAL_COLUMNS if column not in feature_values]
        if missing:
            raise KeyError(f"Missing categorical feature(s): {missing}")

    values = []
    for index, column in enumerate(CATEGORICAL_COLUMNS):
        value = int(feature_values.get(column, 0) if coerce else feature_values[column])
        if coerce:
            value = max(0, value)
            value = min(value, cardinalities[index] - 1)
            values.append(value)
            continue
        if value < 0:
            raise ValueError(f"{column} must be >= 0, got {value}")
        if value >= cardinalities[index]:
            raise ValueError(
                f"{column}={value} is outside trained range [0, {cardinalities[index] - 1}]"
            )
        values.append(value)
    return torch.tensor([values], dtype=torch.long)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))


def _load_cardinalities(processed_dir: Path, device: torch.device) -> list[int]:
    return torch.load(processed_dir / "train_cardinalities.pt", map_location=device).tolist()


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


def load_end_to_end_resources(
    *,
    processed_dir: Path = REPO_ROOT / "dataset" / "processed",
    checkpoint_dir: Path = REPO_ROOT / "checkpoints",
    checkpoint_prefix: str | None = None,
    naverpay_path: Path | None = None,
    pykrx_path: Path | None = None,
    device: str | torch.device | None = None,
    anchor_csv: Path = REPO_ROOT / "dataset" / "splits" / "train.csv",
    load_knn_anchors: bool = True,
    current_x_cat_tensor: torch.Tensor | None = None,
    cardinalities: list[int] | None = None,
) -> EndToEndResources:
    resolved_device = _resolve_device(device)
    resolved_cardinalities = (
        cardinalities if cardinalities is not None else _load_cardinalities(processed_dir, resolved_device)
    )
    resolved_prefix = _resolve_checkpoint_prefix(checkpoint_dir, checkpoint_prefix)

    source_encoder, _, checkpoint_meta, validation = load_dual_encoder_checkpoint(
        checkpoint_dir=checkpoint_dir,
        device=resolved_device,
        current_categorical_cols=CATEGORICAL_COLUMNS,
        current_categorical_cardinalities=resolved_cardinalities,
        current_x_cat_tensor=current_x_cat_tensor,
        current_split=None,
        strict=False,
        prefix=resolved_prefix,
    )

    x_anchor_cat = None
    anchor_allocs = None
    anchor_risky_shares = None
    if load_knn_anchors:
        x_anchor_cat, anchor_allocs, anchor_risky_shares = load_anchor_arrays(
            str(anchor_csv),
            resolved_cardinalities,
        )

    return EndToEndResources(
        source_encoder=source_encoder,
        checkpoint_meta=checkpoint_meta,
        validation=validation,
        cardinalities=resolved_cardinalities,
        products=_load_products(naverpay_path, pykrx_path),
        device=resolved_device,
        checkpoint_prefix=resolved_prefix,
        x_anchor_cat=x_anchor_cat,
        anchor_allocs=anchor_allocs,
        anchor_risky_shares=anchor_risky_shares,
        anchor_csv=anchor_csv,
    )


def run_end_to_end(
    profile: dict,
    *,
    resources: EndToEndResources | None = None,
    input_path: Path | str | None = None,
    processed_dir: Path = REPO_ROOT / "dataset" / "processed",
    checkpoint_dir: Path = REPO_ROOT / "checkpoints",
    checkpoint_prefix: str | None = None,
    naverpay_path: Path | None = None,
    pykrx_path: Path | None = None,
    device: str | torch.device | None = None,
    risk_source: str = "model",
    anchor_csv: Path = REPO_ROOT / "dataset" / "splits" / "train.csv",
    disable_knn_smoothing: bool = False,
    knn_k: int = 20,
    knn_alpha: float = 0.7,
    coerce_input: bool = False,
) -> dict:
    if risk_source not in {"model", "allocation"}:
        raise ValueError("risk_source must be either 'model' or 'allocation'")

    if resources is None:
        resolved_device = _resolve_device(device)
        cardinalities = _load_cardinalities(processed_dir, resolved_device)
        x_cat = _build_input_tensor(profile, cardinalities, coerce=coerce_input)
        resources = load_end_to_end_resources(
            processed_dir=processed_dir,
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=checkpoint_prefix,
            naverpay_path=naverpay_path,
            pykrx_path=pykrx_path,
            device=resolved_device,
            anchor_csv=anchor_csv,
            load_knn_anchors=not disable_knn_smoothing,
            current_x_cat_tensor=x_cat,
            cardinalities=cardinalities,
        )
    else:
        x_cat = _build_input_tensor(profile, resources.cardinalities, coerce=coerce_input)

    with torch.no_grad():
        source_output = resources.source_encoder(x_cat.to(resources.device))

    allocation = source_output.allocation_probs.squeeze(0).cpu().numpy()
    model_risky_share = float(source_output.risky_share.squeeze().cpu().item())
    model_risk_level = risky_share_to_bucket(model_risky_share)
    allocation_risk_level = derive_risk_label_from_allocation_vector(allocation)

    smoothing_payload = None
    final_risky_share = model_risky_share
    if not disable_knn_smoothing:
        x_anchor_cat = resources.x_anchor_cat
        anchor_allocs = resources.anchor_allocs
        anchor_risky_shares = resources.anchor_risky_shares
        if x_anchor_cat is None or anchor_allocs is None or anchor_risky_shares is None:
            anchor_path = resources.anchor_csv if resources.anchor_csv is not None else anchor_csv
            x_anchor_cat, anchor_allocs, anchor_risky_shares = load_anchor_arrays(
                str(anchor_path),
                resources.cardinalities,
            )
        smoothing_result = smooth_with_profile_knn(
            x_user_cat=x_cat.squeeze(0).numpy(),
            x_anchor_cat=x_anchor_cat,
            pred_alloc=allocation,
            anchor_allocs=anchor_allocs,
            pred_risky_share=model_risky_share,
            anchor_risky_shares=anchor_risky_shares,
            cardinalities=resources.cardinalities,
            k=knn_k,
            alpha=knn_alpha,
        )
        allocation = smoothing_result.allocation
        final_risky_share = smoothing_result.risky_share
        smoothing_payload = {
            "enabled": True,
            "k": knn_k,
            "alpha": knn_alpha,
            "anchor_allocation": {
                bucket: float(value)
                for bucket, value in zip(BUCKET_COLUMNS, smoothing_result.anchor_allocation.tolist())
            },
            "anchor_risky_share": smoothing_result.anchor_risky_share,
            "topk_indices": [int(value) for value in smoothing_result.topk_indices.tolist()],
        }
    else:
        smoothing_payload = {"enabled": False}

    model_risk_level = risky_share_to_bucket(final_risky_share)
    allocation_risk_level = derive_risk_label_from_allocation_vector(allocation)

    options = profile.get("options", {})
    top_k = int(options.get("top_k", 5))
    allow_cma = bool(options.get("allow_cma", False))
    default_risk_level = model_risk_level if risk_source == "model" else allocation_risk_level
    risk_level_for_recs = int(options.get("risk_level_override", default_risk_level))

    recommendation = recommend_products(
        resources.products,
        UserRequest(
            risk_level=risk_level_for_recs,
            predicted_allocation=allocation,
            allow_cma=allow_cma,
        ),
        top_k=top_k,
    )

    return {
        "input_path": str(input_path) if input_path is not None else None,
        "checkpoint_prefix": resources.checkpoint_prefix,
        "validation_warnings": resources.validation["warnings"],
        "categorical_features": {
            column: int(value)
            for column, value in zip(CATEGORICAL_COLUMNS, x_cat.squeeze(0).tolist())
        },
        "predicted_allocation": {
            bucket: float(value) for bucket, value in zip(BUCKET_COLUMNS, allocation.tolist())
        },
        "predicted_risky_share": final_risky_share,
        "predicted_risk_level_model": model_risk_level,
        "predicted_risk_level_allocation": allocation_risk_level,
        "risk_level_used_for_recommendation": risk_level_for_recs,
        "risk_source_used_for_recommendation": risk_source,
        "knn_smoothing": smoothing_payload,
        "options": {
            "top_k": top_k,
            "allow_cma": allow_cma,
        },
        "recommendation": recommendation,
        "model_config": resources.checkpoint_meta["model_config"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Path to user profile JSON")
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "dataset" / "processed")
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--checkpoint-prefix", type=str, default=None)
    parser.add_argument("--naverpay-path", type=Path, default=None)
    parser.add_argument("--pykrx-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--risk-source", choices=["model", "allocation"], default="model")
    parser.add_argument("--anchor-csv", type=Path, default=REPO_ROOT / "dataset" / "splits" / "train.csv")
    parser.add_argument("--disable-knn-smoothing", action="store_true")
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--knn-alpha", type=float, default=0.7)
    args = parser.parse_args()

    profile = _load_profile(args.input)
    result = run_end_to_end(
        profile,
        input_path=args.input,
        processed_dir=args.processed_dir,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_prefix=args.checkpoint_prefix,
        naverpay_path=args.naverpay_path,
        pykrx_path=args.pykrx_path,
        device=args.device,
        risk_source=args.risk_source,
        anchor_csv=args.anchor_csv,
        disable_knn_smoothing=args.disable_knn_smoothing,
        knn_k=args.knn_k,
        knn_alpha=args.knn_alpha,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
