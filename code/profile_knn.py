from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import OneHotEncoder

from portfolio_schema import build_allocation_dataset


@dataclass
class ProfileKnnResult:
    allocation: np.ndarray
    risky_share: float
    topk_indices: np.ndarray
    anchor_allocation: np.ndarray
    anchor_risky_share: float


def _build_one_hot_encoder(anchor_x_cat: np.ndarray, cardinalities: list[int]) -> OneHotEncoder:
    categories = [np.arange(cardinality) for cardinality in cardinalities]
    try:
        return OneHotEncoder(categories=categories, sparse_output=False, handle_unknown="ignore").fit(anchor_x_cat)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(categories=categories, sparse=False, handle_unknown="ignore").fit(anchor_x_cat)


def smooth_with_profile_knn(
    x_user_cat: np.ndarray,
    x_anchor_cat: np.ndarray,
    pred_alloc: np.ndarray,
    anchor_allocs: np.ndarray,
    pred_risky_share: float,
    anchor_risky_shares: np.ndarray,
    cardinalities: list[int],
    *,
    k: int = 20,
    alpha: float = 0.7,
) -> ProfileKnnResult:
    if len(x_anchor_cat) == 0:
        raise ValueError("At least one anchor profile is required for kNN smoothing.")

    k = max(1, min(int(k), len(x_anchor_cat)))
    alpha = float(np.clip(alpha, 0.0, 1.0))

    encoder = _build_one_hot_encoder(x_anchor_cat, cardinalities)
    user_onehot = encoder.transform(x_user_cat.reshape(1, -1))
    anchor_onehot = encoder.transform(x_anchor_cat)
    distances = cosine_distances(user_onehot, anchor_onehot)[0]
    topk_idx = np.argsort(distances)[:k]

    anchor_mean = anchor_allocs[topk_idx].mean(axis=0)
    smoothed_alloc = alpha * pred_alloc + (1.0 - alpha) * anchor_mean
    smoothed_alloc = np.clip(smoothed_alloc, 1e-8, None)
    smoothed_alloc = smoothed_alloc / smoothed_alloc.sum()

    anchor_risky_mean = float(anchor_risky_shares[topk_idx].mean())
    smoothed_risky = alpha * float(pred_risky_share) + (1.0 - alpha) * anchor_risky_mean
    smoothed_risky = float(np.clip(smoothed_risky, 0.0, 1.0))

    return ProfileKnnResult(
        allocation=smoothed_alloc.astype(np.float32),
        risky_share=smoothed_risky,
        topk_indices=topk_idx,
        anchor_allocation=anchor_mean.astype(np.float32),
        anchor_risky_share=anchor_risky_mean,
    )


def load_anchor_arrays(anchor_csv: str, cardinalities: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(anchor_csv)
    build_result = build_allocation_dataset(frame)
    x_anchor_cat = build_result.categorical_frame.values.astype(np.int64)
    anchor_allocs = build_result.allocation_frame.values.astype(np.float32)
    anchor_risky_shares = build_result.risky_share.values.astype(np.float32)

    for col_idx, cardinality in enumerate(cardinalities):
        x_anchor_cat[:, col_idx] = np.clip(x_anchor_cat[:, col_idx], 0, cardinality - 1)

    return x_anchor_cat, anchor_allocs, anchor_risky_shares
