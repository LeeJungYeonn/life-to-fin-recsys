from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .product_schema import Product, ProductExposure


def classify_etf_exposure(name: str) -> ProductExposure:
    lowered = name.lower()
    if any(token in name for token in ["국고채", "단기채", "통안채", "회사채", "채권", "bond"]):
        return ProductExposure(taxable_bond=1.0)
    if any(token in name for token in ["파킹", "머니마켓", "mmf"]):
        return ProductExposure(cash_eq=1.0)
    if any(token in name for token in ["연금", "은퇴", "target date", "tdf"]):
        return ProductExposure(retirement_equity=0.6, retirement_safe=0.4)
    if any(token in lowered for token in ["high dividend", "dividend", "고배당"]):
        return ProductExposure(taxable_equity=0.8, taxable_bond=0.2)
    return ProductExposure(taxable_equity=1.0)


def build_products_from_snapshot(snapshot: pd.DataFrame) -> List[Product]:
    products = []
    for row in snapshot.to_dict(orient="records"):
        name = str(row["Name"])
        products.append(
            Product(
                product_id=f"pykrx:{row['Ticker']}",
                source="pykrx",
                provider=str(row.get("Provider", "ETF")),
                name=name,
                product_type="etf",
                category=str(row.get("Risk_Level", "unknown")),
                principal_protection=False,
                deposit_insurance=False,
                base_rate=None,
                max_rate=None,
                volatility=float(row["Volatility(%)"]) if "Volatility(%)" in row and pd.notna(row["Volatility(%)"]) else None,
                liquidity_tier="high",
                exposure=classify_etf_exposure(name),
                tags=[tag for tag in [row.get("Market"), row.get("Theme")] if isinstance(tag, str) and tag],
                metadata={"ticker": row["Ticker"]},
            )
        )
    return products


def load_snapshot(path: Path) -> List[Product]:
    frame = pd.read_csv(path)
    return build_products_from_snapshot(frame)


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
