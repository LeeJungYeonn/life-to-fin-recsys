from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np


def normalized_hhi_diversification(
    weights: Sequence[float],
    groups: Sequence[object],
    *,
    possible_groups: Iterable[object] | None = None,
    max_groups: int | None = None,
) -> float:
    """Return 0-1 diversification from group concentration using normalized HHI."""
    if len(weights) != len(groups):
        raise ValueError("weights and groups must have the same length.")

    raw_weights = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    positive_mask = raw_weights > 0.0
    if raw_weights.sum() <= 0.0 or not np.any(positive_mask):
        return 0.0

    group_weights: dict[object, float] = defaultdict(float)
    for weight, group in zip(raw_weights[positive_mask], np.asarray(groups, dtype=object)[positive_mask]):
        group_weights[group] += float(weight)

    if possible_groups is None:
        group_count = len(group_weights)
    else:
        group_count = len(set(possible_groups))
    if max_groups is not None:
        group_count = min(group_count, max_groups)
    if group_count <= 1:
        return 0.0

    proportions = np.asarray(list(group_weights.values()), dtype=np.float64)
    proportions = proportions / proportions.sum()
    hhi = float(np.sum(proportions**2))
    score = (1.0 - hhi) / (1.0 - (1.0 / group_count))
    return float(np.clip(score, 0.0, 1.0))


def allocation_diversification_scores(
    allocations: np.ndarray,
    bucket_columns: Sequence[str],
) -> np.ndarray:
    allocations = np.asarray(allocations, dtype=np.float64)
    if allocations.ndim == 1:
        allocations = allocations.reshape(1, -1)
    return np.asarray(
        [
            normalized_hhi_diversification(row, bucket_columns, possible_groups=bucket_columns)
            for row in allocations
        ],
        dtype=np.float64,
    )


def basket_diversification_score(
    basket: Sequence[Mapping[str, object]],
    group_key: str,
    *,
    possible_groups: Iterable[object] | None = None,
) -> float:
    if not basket:
        return 0.0

    weights = []
    groups = []
    for item in basket:
        weights.append(float(item.get("weight", 0.0) or 0.0))
        groups.append(item.get(group_key) or "unknown")

    positive_items = int(np.sum(np.asarray(weights, dtype=np.float64) > 0.0))
    return normalized_hhi_diversification(
        weights,
        groups,
        possible_groups=possible_groups,
        max_groups=max(positive_items, 1),
    )


def basket_diversification_scores(
    basket: Sequence[Mapping[str, object]],
    *,
    possible_buckets: Iterable[object],
    possible_categories: Iterable[object],
    possible_subtypes: Iterable[object],
    asset_weight: float = 0.4,
) -> dict[str, float]:
    asset_score = basket_diversification_score(
        basket,
        "bucket",
        possible_groups=possible_buckets,
    )
    category_score = basket_diversification_score(
        basket,
        "category",
        possible_groups=possible_categories,
    )
    subtype_score = basket_diversification_score(
        basket,
        "subtype",
        possible_groups=possible_subtypes,
    )
    overall = asset_weight * asset_score + (1.0 - asset_weight) * subtype_score
    return {
        "asset_diversification": float(asset_score),
        "category_diversification": float(category_score),
        "subtype_diversification": float(subtype_score),
        "overall_diversification": float(overall),
    }
