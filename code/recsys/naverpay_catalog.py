from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .product_schema import Product, ProductExposure


NAVERPAY_CATEGORY_URLS = {
    "deposit": "https://pay.naver.com/savings/list/deposit",
    "saving": "https://pay.naver.com/savings/list/saving",
    "parking": "https://pay.naver.com/savings/list/parking",
    "cma": "https://pay.naver.com/savings/list/cma",
}

DEFAULT_SNAPSHOT_CANDIDATES = [
    Path("../dataset/catalogs/naverpay_products.json"),
    Path("../dataset/catalogs/naverpay_products_template.json"),
    Path("dataset/catalogs/naverpay_products.json"),
    Path("dataset/catalogs/naverpay_products_template.json"),
]


def _default_exposure(category: str) -> ProductExposure:
    if category == "cma":
        return ProductExposure(cash_eq=1.0)
    if category == "parking":
        return ProductExposure(cash_eq=1.0)
    if category == "deposit":
        return ProductExposure(cash_eq=0.9, retirement_safe=0.1)
    if category == "saving":
        return ProductExposure(cash_eq=0.8, retirement_safe=0.2)
    return ProductExposure(other_fin=1.0)


def load_snapshot(path: Path) -> List[Product]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    products = []
    for item in payload:
        category = item["category"]
        principal_protection = bool(item.get("principal_protection", category != "cma"))
        deposit_insurance = bool(item.get("deposit_insurance", category != "cma"))
        products.append(
            Product(
                product_id=str(item["product_id"]),
                source="naverpay",
                provider=str(item["provider"]),
                name=str(item["name"]),
                product_type=str(item.get("product_type", category)),
                category=category,
                principal_protection=principal_protection,
                deposit_insurance=deposit_insurance,
                base_rate=float(item["base_rate"]) if item.get("base_rate") is not None else None,
                max_rate=float(item["max_rate"]) if item.get("max_rate") is not None else None,
                liquidity_tier=str(item.get("liquidity_tier", "high")),
                min_term_months=item.get("min_term_months"),
                max_term_months=item.get("max_term_months"),
                exposure=_default_exposure(category),
                tags=list(item.get("tags", [])),
                metadata={
                    "product_url": item.get("product_url", NAVERPAY_CATEGORY_URLS.get(category)),
                    "source_note": item.get("source_note", "naverpay snapshot"),
                },
            )
        )
    return products


def load_default_snapshot() -> List[Product]:
    for candidate in DEFAULT_SNAPSHOT_CANDIDATES:
        if candidate.exists():
            return load_snapshot(candidate)
    raise FileNotFoundError(
        "Naver Pay snapshot file not found. Expected one of: "
        + ", ".join(str(path) for path in DEFAULT_SNAPSHOT_CANDIDATES)
    )
