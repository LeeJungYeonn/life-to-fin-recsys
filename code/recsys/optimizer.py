from __future__ import annotations

from typing import Dict, List

import numpy as np

from .product_schema import BUCKET_COLUMNS, Product


def _project_simplex(weights: np.ndarray) -> np.ndarray:
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    if total <= 0.0:
        return np.full_like(weights, 1.0 / len(weights))
    return weights / total


def _dominant_bucket(product: Product) -> str:
    exposure = np.array(product.exposure_vector(), dtype=np.float32)
    if exposure.sum() <= 0.0:
        return BUCKET_COLUMNS[0]
    return BUCKET_COLUMNS[int(np.argmax(exposure))]


def _allocate_slots(target_allocation: np.ndarray, top_k: int) -> np.ndarray:
    target = _project_simplex(np.asarray(target_allocation, dtype=np.float32))
    raw = target * top_k
    slots = np.floor(raw).astype(int)
    positive = target > 0.0
    slots[(slots == 0) & positive] = 1
    while slots.sum() > top_k:
        candidates = np.where(slots > 0)[0]
        idx = candidates[np.argmin(raw[candidates] - np.floor(raw[candidates]))]
        slots[idx] -= 1
    while slots.sum() < top_k:
        idx = int(np.argmax(raw - slots))
        slots[idx] += 1
    return slots


def _select_allocation_aware_products(
    target_allocation: np.ndarray,
    products: List[Product],
    scores: Dict[str, float],
    top_k: int,
) -> List[Product]:
    selected: list[Product] = []
    selected_ids: set[str] = set()
    slots = _allocate_slots(target_allocation, top_k)

    by_bucket = {bucket: [] for bucket in BUCKET_COLUMNS}
    for product in products:
        by_bucket[_dominant_bucket(product)].append(product)
    for bucket_products in by_bucket.values():
        bucket_products.sort(key=lambda product: scores.get(product.product_id, 0.0), reverse=True)

    for bucket_idx, bucket in enumerate(BUCKET_COLUMNS):
        for product in by_bucket[bucket]:
            if len([p for p in selected if _dominant_bucket(p) == bucket]) >= slots[bucket_idx]:
                break
            if product.product_id not in selected_ids:
                selected.append(product)
                selected_ids.add(product.product_id)

    for product in sorted(products, key=lambda item: scores.get(item.product_id, 0.0), reverse=True):
        if len(selected) >= top_k:
            break
        if product.product_id not in selected_ids:
            selected.append(product)
            selected_ids.add(product.product_id)

    return selected[:top_k]


def optimize_product_mix(
    target_allocation: np.ndarray,
    products: List[Product],
    scores: Dict[str, float],
    *,
    top_k: int = 5,
    max_weight: float = 0.7,
) -> List[dict]:
    ranked = _select_allocation_aware_products(target_allocation, products, scores, top_k)
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
