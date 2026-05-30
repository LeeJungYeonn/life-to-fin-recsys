from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


BUCKET_COLUMNS = [
    "cash",
    "bond",
    "pension",
    "equity",
]


@dataclass
class ProductExposure:
    cash: float = 0.0
    bond: float = 0.0
    pension: float = 0.0
    equity: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {bucket: float(getattr(self, bucket)) for bucket in BUCKET_COLUMNS}


@dataclass
class Product:
    product_id: str
    source: str
    provider: str
    name: str
    product_type: str
    category: str
    principal_protection: bool
    deposit_insurance: bool
    base_rate: float | None = None
    max_rate: float | None = None
    volatility: float | None = None
    liquidity_tier: str = "medium"
    min_term_months: int | None = None
    max_term_months: int | None = None
    exposure: ProductExposure = field(default_factory=ProductExposure)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def exposure_vector(self) -> List[float]:
        payload = self.exposure.as_dict()
        return [payload[key] for key in BUCKET_COLUMNS]
