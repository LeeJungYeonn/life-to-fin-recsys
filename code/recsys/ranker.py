from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from .graph_builder import bucket_affinity, diffuse_product_scores
from .optimizer import optimize_product_mix
from .product_schema import Product


@dataclass
class UserRequest:
    risk_level: int
    predicted_allocation: np.ndarray
    allow_cma: bool = False


def _filter_products(products: Iterable[Product], request: UserRequest) -> List[Product]:
    filtered = []
    for product in products:
        if request.risk_level <= 1:
            if product.category == "cma" and not request.allow_cma:
                continue
            if product.product_type == "etf":
                continue
        if request.risk_level <= 2 and product.category == "cma" and not request.allow_cma:
            continue
        filtered.append(product)
    return filtered


def _score_product(product: Product, request: UserRequest) -> float:
    allocation_match = bucket_affinity(request.predicted_allocation, product)
    yield_bonus = 0.0
    if product.max_rate is not None:
        yield_bonus += 0.05 * product.max_rate
    if product.base_rate is not None:
        yield_bonus += 0.03 * product.base_rate
    safety_bonus = 0.2 if product.principal_protection else -0.3
    insurance_bonus = 0.1 if product.deposit_insurance else -0.15
    liquidity_bonus = 0.1 if product.liquidity_tier == "high" else 0.0
    cma_penalty = 0.25 if product.category == "cma" and request.risk_level <= 1 and not request.allow_cma else 0.0
    return allocation_match + yield_bonus + safety_bonus + insurance_bonus + liquidity_bonus - cma_penalty


def recommend_products(products: Iterable[Product], request: UserRequest, top_k: int = 5) -> Dict[str, object]:
    candidate_products = _filter_products(products, request)
    base_scores = {product.product_id: _score_product(product, request) for product in candidate_products}
    graph_scores = diffuse_product_scores(base_scores, candidate_products)
    basket = optimize_product_mix(
        request.predicted_allocation,
        candidate_products,
        graph_scores,
        top_k=top_k,
    )
    ranked = sorted(
        (
            {
                "product_id": product.product_id,
                "name": product.name,
                "score": float(graph_scores.get(product.product_id, 0.0)),
                "category": product.category,
                "principal_protection": product.principal_protection,
                "deposit_insurance": product.deposit_insurance,
            }
            for product in candidate_products
        ),
        key=lambda item: item["score"],
        reverse=True,
    )[:top_k]
    return {
        "ranked_products": ranked,
        "optimized_basket": basket,
    }
