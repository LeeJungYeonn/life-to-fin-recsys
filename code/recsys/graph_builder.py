from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np

from .product_schema import BUCKET_COLUMNS, Product


def build_product_similarity(products: Iterable[Product]) -> Dict[str, Dict[str, float]]:
    product_list = list(products)
    vectors = {
        product.product_id: np.array(product.exposure_vector(), dtype=np.float32)
        for product in product_list
    }
    graph = defaultdict(dict)
    for left in product_list:
        for right in product_list:
            if left.product_id == right.product_id:
                continue
            left_vec = vectors[left.product_id]
            right_vec = vectors[right.product_id]
            denom = np.linalg.norm(left_vec) * np.linalg.norm(right_vec)
            cosine = float(np.dot(left_vec, right_vec) / denom) if denom > 0 else 0.0
            graph[left.product_id][right.product_id] = max(cosine, 0.0)
    return graph


def bucket_affinity(target_allocation: np.ndarray, product: Product) -> float:
    product_vec = np.array(product.exposure_vector(), dtype=np.float32)
    return float(np.dot(target_allocation, product_vec))


def diffuse_product_scores(
    base_scores: Dict[str, float],
    products: List[Product],
    alpha: float = 0.85,
    steps: int = 5,
) -> Dict[str, float]:
    similarity = build_product_similarity(products)
    scores = dict(base_scores)
    for _ in range(steps):
        next_scores = {}
        for product in products:
            neighbor_score = 0.0
            for neighbor_id, weight in similarity[product.product_id].items():
                neighbor_score += scores.get(neighbor_id, 0.0) * weight
            next_scores[product.product_id] = (1 - alpha) * base_scores.get(product.product_id, 0.0) + alpha * neighbor_score
        scores = next_scores
    return scores
