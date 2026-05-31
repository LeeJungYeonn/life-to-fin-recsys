from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .product_schema import Product, ProductExposure

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SNAPSHOT_CANDIDATES = [
    REPO_ROOT / "dataset" / "catalogs" / "pykrx_etf_snapshot.csv",
]


def _first_present(row: dict, *columns: str):
    for column in columns:
        value = row.get(column)
        if pd.notna(value):
            return value
    return None


def exposure_from_asset_class(asset_class: str | None) -> ProductExposure | None:
    if asset_class is None:
        return None
    normalized = str(asset_class).strip().lower()
    if normalized == "cash":
        return ProductExposure(cash=1.0)
    if normalized == "bond":
        return ProductExposure(bond=1.0)
    if normalized == "pension":
        return ProductExposure(pension=1.0)
    if normalized == "equity":
        return ProductExposure(equity=1.0)
    return None


def classify_etf_exposure(name: str) -> ProductExposure:
    lowered = name.lower()
    if any(token in name for token in ["국고채", "단기채", "통안채", "회사채", "채권", "bond"]):
        return ProductExposure(bond=1.0)
    if any(token in name for token in ["파킹", "머니마켓", "mmf"]):
        return ProductExposure(cash=1.0)
    if any(token in name for token in ["연금", "은퇴", "target date", "tdf"]):
        return ProductExposure(pension=1.0)
    if any(token in lowered for token in ["high dividend", "dividend", "고배당"]):
        return ProductExposure(equity=0.8, bond=0.2)
    return ProductExposure(equity=1.0)


def build_products_from_snapshot(snapshot: pd.DataFrame) -> List[Product]:
    products = []
    for row in snapshot.to_dict(orient="records"):
        name = str(_first_present(row, "Name", "product_name") or row["Ticker"])
        risk_level = _first_present(row, "Risk_Level", "risk_level")
        category = str(int(risk_level)) if pd.notna(risk_level) else "unknown"
        provider = str(_first_present(row, "Provider", "provider") or name.split()[0] or "ETF")
        asset_class = _first_present(row, "asset_class")
        subtype = _first_present(row, "subtype", "Theme")
        exposure = exposure_from_asset_class(str(asset_class) if asset_class is not None else None)
        if exposure is None:
            exposure = classify_etf_exposure(name)
        products.append(
            Product(
                product_id=f"pykrx:{_first_present(row, 'Ticker', 'product_id')}",
                source="pykrx",
                provider=provider,
                name=name,
                product_type="etf",
                category=category,
                principal_protection=False,
                deposit_insurance=False,
                base_rate=None,
                max_rate=None,
                volatility=float(_first_present(row, "Volatility(%)", "volatility"))
                if _first_present(row, "Volatility(%)", "volatility") is not None
                else None,
                liquidity_tier="high",
                exposure=exposure,
                tags=[
                    tag
                    for tag in [
                        _first_present(row, "Market", "market"),
                        subtype,
                        str(asset_class) if asset_class is not None else None,
                        f"risk_{category}" if category != "unknown" else None,
                    ]
                    if isinstance(tag, str) and tag
                ],
                metadata={
                    "ticker": _first_present(row, "Ticker", "product_id"),
                    "asset_class": str(asset_class) if asset_class is not None else None,
                    "subtype": str(subtype) if subtype is not None else None,
                    "risk_level": int(risk_level) if pd.notna(risk_level) else None,
                    "preference_score": float(_first_present(row, "Preference_Score", "score"))
                    if _first_present(row, "Preference_Score", "score") is not None
                    else None,
                    "theme_risk_level": int(_first_present(row, "Theme_Risk_Level", "theme_risk_level"))
                    if _first_present(row, "Theme_Risk_Level", "theme_risk_level") is not None
                    else None,
                },
            )
        )
    return products


def load_snapshot(path: Path) -> List[Product]:
    frame = pd.read_csv(path)
    return build_products_from_snapshot(frame)


def find_default_snapshot_path() -> Path | None:
    for candidate in DEFAULT_SNAPSHOT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def load_default_snapshot() -> List[Product]:
    path = find_default_snapshot_path()
    if path is None:
        raise FileNotFoundError(
            "pykrx ETF snapshot file not found. Expected one of: "
            + ", ".join(str(path) for path in DEFAULT_SNAPSHOT_CANDIDATES)
        )
    return load_snapshot(path)


def try_fetch_live_snapshot(
    *,
    as_of: str,
    tickers: Iterable[str] | None = None,
    max_items: int | None = None,
) -> List[Product]:
    from pykrx import stock

    all_tickers = list(tickers or stock.get_etf_ticker_list(as_of))
    if max_items is not None:
        all_tickers = all_tickers[:max_items]

    rows = []
    for ticker in all_tickers:
        name = stock.get_etf_ticker_name(ticker)
        rows.append({"Ticker": ticker, "Name": name})
    return build_products_from_snapshot(pd.DataFrame(rows))
