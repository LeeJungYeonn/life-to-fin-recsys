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

    by_bucket = {bucket: [] for bucket in BUCKET_COLUMNS}
    for product in products:
        by_bucket[_dominant_bucket(product)].append(product)
    for bucket_products in by_bucket.values():
        bucket_products.sort(key=lambda product: scores.get(product.product_id, 0.0), reverse=True)

    target = _project_simplex(np.asarray(target_allocation, dtype=np.float32))
    available_target = target.copy()
    for bucket_idx, bucket in enumerate(BUCKET_COLUMNS):
        if not by_bucket[bucket]:
            available_target[bucket_idx] = 0.0
    slots = _allocate_slots(available_target, top_k)

    for bucket_idx, bucket in enumerate(BUCKET_COLUMNS):
        for product in by_bucket[bucket]:
            if len([p for p in selected if _dominant_bucket(p) == bucket]) >= slots[bucket_idx]:
                break
            if product.product_id not in selected_ids:
                selected.append(product)
                selected_ids.add(product.product_id)

    target_buckets = {
        bucket
        for bucket_idx, bucket in enumerate(BUCKET_COLUMNS)
        if available_target[bucket_idx] > 0.0
    }
    for product in sorted(products, key=lambda item: scores.get(item.product_id, 0.0), reverse=True):
        if len(selected) >= top_k:
            break
        if _dominant_bucket(product) in target_buckets and product.product_id not in selected_ids:
            selected.append(product)
            selected_ids.add(product.product_id)

    return selected[:top_k]


def _bucket_score_weights(products: List[Product], scores: Dict[str, float]) -> np.ndarray:
    score_values = np.array(
        [max(scores.get(product.product_id, 0.0), 0.0) for product in products],
        dtype=np.float32,
    )
    if score_values.sum() <= 0.0:
        return np.full(len(products), 1.0 / len(products), dtype=np.float32)
    return score_values / score_values.sum()


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

    target = _project_simplex(np.asarray(target_allocation, dtype=np.float32))
    selected_buckets = {_dominant_bucket(product) for product in ranked}
    selected_target = np.array(
        [
            target[bucket_idx] if bucket in selected_buckets else 0.0
            for bucket_idx, bucket in enumerate(BUCKET_COLUMNS)
        ],
        dtype=np.float32,
    )
    selected_target = _project_simplex(selected_target)

    weights_by_id = {}
    target_by_id = {}
    bucket_by_id = {}
    for bucket_idx, bucket in enumerate(BUCKET_COLUMNS):
        bucket_products = [product for product in ranked if _dominant_bucket(product) == bucket]
        if not bucket_products:
            continue
        bucket_weight = float(selected_target[bucket_idx])
        in_bucket_weights = _bucket_score_weights(bucket_products, scores)
        for product, weight in zip(bucket_products, in_bucket_weights):
            weights_by_id[product.product_id] = bucket_weight * float(weight)
            target_by_id[product.product_id] = bucket_weight
            bucket_by_id[product.product_id] = bucket

    return [
        {
            "product_id": product.product_id,
            "name": product.name,
            "weight": float(weights_by_id[product.product_id]),
            "score": float(scores.get(product.product_id, 0.0)),
            "category": product.category,
            "bucket": bucket_by_id[product.product_id],
            "target_bucket_weight": float(target_by_id[product.product_id]),
        }
        for product in ranked
    ]
