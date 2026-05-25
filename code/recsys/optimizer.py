from __future__ import annotations

from typing import Dict, List

import numpy as np

from .product_schema import Product


def _project_simplex(weights: np.ndarray) -> np.ndarray:
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    if total <= 0.0:
        return np.full_like(weights, 1.0 / len(weights))
    return weights / total


def optimize_product_mix(
    target_allocation: np.ndarray,
    products: List[Product],
    scores: Dict[str, float],
    *,
    top_k: int = 5,
    max_weight: float = 0.7,
) -> List[dict]:
    ranked = sorted(products, key=lambda product: scores.get(product.product_id, 0.0), reverse=True)[:top_k]
    if not ranked:
        return []

    exposure_matrix = np.array([product.exposure_vector() for product in ranked], dtype=np.float32).T
    score_prior = np.array([max(scores.get(product.product_id, 0.0), 0.0) for product in ranked], dtype=np.float32)
    if score_prior.sum() <= 0.0:
        score_prior = np.ones_like(score_prior)
    score_prior = score_prior / score_prior.sum()

    raw_weights, *_ = np.linalg.lstsq(exposure_matrix, target_allocation, rcond=None)
    raw_weights = np.clip(raw_weights, 0.0, max_weight)
    weights = _project_simplex(0.7 * raw_weights + 0.3 * score_prior)

    return [
        {
            "product_id": product.product_id,
            "name": product.name,
            "weight": float(weight),
            "score": float(scores.get(product.product_id, 0.0)),
            "category": product.category,
        }
        for product, weight in zip(ranked, weights)
    ]
