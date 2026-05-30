from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


ASSET_CLASSES = ["cash", "bond", "pension", "equity"]

PROVIDER_PREFIXES = [
    "KODEX",
    "TIGER",
    "ACE",
    "SOL",
    "KBSTAR",
    "RISE",
    "PLUS",
    "ARIRANG",
    "KOSEF",
    "HANARO",
    "TIMEFOLIO",
    "FOCUS",
    "BNK",
    "WOORI",
    "HK",
    "MASTER",
    "마이다스",
]

GLOBAL_MARKET_KEYWORDS = [
    "미국",
    "중국",
    "일본",
    "유럽",
    "인도",
    "베트남",
    "글로벌",
    "선진국",
    "신흥국",
    "WORLD",
    "GLOBAL",
    "MSCI",
    "S&P",
    "NASDAQ",
    "NIKKEI",
    "EURO",
    "HONG KONG",
    "TAIWAN",
    "BRAZIL",
]

KEYWORDS = {
    "cash": [
        "머니마켓",
        "MMF",
        "CD금리",
        "CD 금리",
        "KOFR",
        "SOFR",
        "파킹",
        "초단기",
        "양도성예금증서",
    ],
    "bond": [
        "단기채",
        "국고채",
        "국채",
        "채권",
        "통안채",
        "회사채",
        "특수채",
        "미국채",
        "종합채권",
        "국공채",
        "금융채",
        "AA",
        "AAA",
        "BOND",
    ],
    "pension": [
        "TDF",
        "TRF",
        "퇴직",
        "연금",
        "은퇴",
        "타겟데이트",
        "TARGET DATE",
    ],
    "equity": [
        "KOSPI",
        "KOSDAQ",
        "코스피",
        "코스닥",
        "200",
        "S&P",
        "나스닥",
        "NASDAQ",
        "MSCI",
        "배당",
        "고배당",
        "배당성장",
        "반도체",
        "2차전지",
        "바이오",
        "AI",
        "기술",
        "테크",
        "혁신",
        "성장",
        "주주가치",
        "밸류",
        "헬스케어",
        "방산",
        "모빌리티",
        "자동차",
        "소비",
        "금융",
        "리츠",
        "로봇",
        "TOP10",
        "미국",
        "글로벌",
        "인도",
        "중국",
        "일본",
        "유럽",
    ],
}

SUBTYPE_KEYWORDS = {
    "cash_like": [
        "CD금리",
        "CD 금리",
        "KOFR",
        "SOFR",
        "머니마켓",
        "MMF",
        "파킹",
        "초단기",
        "양도성예금증서",
    ],
    "retirement": ["TDF", "TRF", "퇴직", "연금", "은퇴", "타겟데이트", "TARGET DATE"],
    "short_bond": ["단기채", "통안채", "초단기", "단기", "1년", "3개월"],
    "gov_bond": ["국고채", "국채", "미국채", "국공채"],
    "credit_bond": ["회사채", "금융채", "특수채", "AA", "AAA"],
    "aggregate_bond": ["종합채권", "채권혼합", "채권"],
    "dividend": ["배당", "고배당", "배당성장", "커버드콜", "DIVIDEND"],
    "broad_index": [
        "KOSPI",
        "KOSDAQ",
        "코스피",
        "코스닥",
        "200",
        "S&P500",
        "S&P 500",
        "나스닥100",
        "NASDAQ100",
        "NASDAQ 100",
        "MSCI",
    ],
    "theme_growth": [
        "반도체",
        "2차전지",
        "바이오",
        "AI",
        "기술",
        "테크",
        "혁신",
        "성장",
        "주주가치",
        "밸류",
        "헬스케어",
        "방산",
        "모빌리티",
        "자동차",
        "소비",
        "인공지능",
        "로봇",
        "전력",
        "인프라",
        "메타버스",
        "게임",
        "테마",
    ],
}

THEME_RISK_RULES = {
    5: ["레버리지", "인버스", "2X", "3X", "바이오", "2차전지"],
    4: ["200", "S&P500", "S&P 500", "나스닥100", "NASDAQ100"],
    3: ["고배당", "배당성장", "커버드콜"],
    2: ["국고채", "단기채", "통안채", "특수채", "지방채", "달러단기", "AA", "AAA"],
}

ROLE_SCORE = {
    "cash_like": 0.95,
    "short_bond": 0.9,
    "gov_bond": 0.85,
    "credit_bond": 0.72,
    "aggregate_bond": 0.75,
    "retirement": 0.8,
    "broad_index": 0.92,
    "dividend": 0.78,
    "theme_growth": 0.55,
    "other": 0.35,
}


def resolve_latest_trading_date(lookback_days: int = 10) -> tuple[str, list[str]]:
    from pykrx import stock

    today = datetime.today()
    for offset in range(lookback_days):
        date_str = (today - timedelta(days=offset)).strftime("%Y%m%d")
        try:
            tickers = list(stock.get_etf_ticker_list(date_str))
        except Exception:
            continue
        if tickers:
            return date_str, tickers
    raise ValueError(f"최근 {lookback_days}일 안에서 ETF 거래일을 찾지 못했습니다.")


def scalar_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.Series):
        value = value.dropna()
        return "" if value.empty else str(value.iloc[0])
    if isinstance(value, pd.DataFrame):
        stacked = value.stack().dropna()
        return "" if stacked.empty else str(stacked.iloc[0])
    if isinstance(value, np.ndarray):
        flat = value.ravel()
        return "" if len(flat) == 0 else scalar_text(flat[0])
    if isinstance(value, (list, tuple)):
        return "" if not value else scalar_text(value[0])
    if pd.isna(value):
        return ""
    return str(value)


def normalize_name(name: object) -> str:
    return scalar_text(name).upper().replace(" ", "")


def has_any(normalized_name: str, keywords: Iterable[str]) -> bool:
    return any(keyword.upper().replace(" ", "") in normalized_name for keyword in keywords)


def infer_provider(name: str) -> str:
    normalized = normalize_name(name)
    for provider in PROVIDER_PREFIXES:
        if normalized.startswith(provider.upper().replace(" ", "")):
            return provider
    tokens = str(name).split()
    return tokens[0] if tokens else "ETF"


def infer_market(name: str) -> str:
    normalized = normalize_name(name)
    if has_any(normalized, GLOBAL_MARKET_KEYWORDS):
        return "global"
    return "domestic"


def classify_asset_class(name: str) -> str:
    normalized = normalize_name(name)
    for asset_class in ["cash", "pension", "bond", "equity"]:
        if has_any(normalized, KEYWORDS[asset_class]):
            return asset_class
    return "other"


def classify_subtype(name: str) -> str:
    normalized = normalize_name(name)
    for subtype, keywords in SUBTYPE_KEYWORDS.items():
        if has_any(normalized, keywords):
            return subtype
    return "other"


def infer_theme(name: str, asset_class: str, subtype: str) -> str:
    normalized = normalize_name(name)
    if asset_class in {"cash", "bond", "pension"}:
        return subtype
    if has_any(normalized, ["나스닥100", "NASDAQ100", "NASDAQ 100"]):
        return "nasdaq100"
    if has_any(normalized, ["S&P500", "S&P 500"]):
        return "sp500"
    if has_any(normalized, ["코스피200", "KOSPI200", "200"]):
        return "index200"
    if has_any(normalized, ["반도체", "SEMICONDUCTOR"]):
        return "semiconductor"
    if has_any(normalized, ["2차전지", "BATTERY"]):
        return "battery"
    if has_any(normalized, ["바이오"]):
        return "bio"
    if subtype == "dividend":
        return "dividend"
    if subtype == "theme_growth":
        return "theme_growth"
    return "equity"


def infer_theme_risk_level(name: str) -> int | None:
    normalized = normalize_name(name)
    for level in sorted(THEME_RISK_RULES.keys(), reverse=True):
        if has_any(normalized, THEME_RISK_RULES[level]):
            return level
    return None


def get_exclude_reason(name: str) -> str | None:
    normalized = normalize_name(name)
    if has_any(normalized, ["레버리지", "2X", "3X", "인버스", "곱버스"]):
        return "leveraged_or_inverse"
    if has_any(normalized, ["합성", "파생", "스왑"]):
        return "synthetic_or_derivative"
    return None


def infer_expected_role(asset_class: str, subtype: str) -> str:
    if asset_class == "cash":
        return "liquidity_buffer"
    if asset_class == "bond" and subtype in {"cash_like", "short_bond"}:
        return "stable_bond"
    if asset_class == "bond" and subtype == "gov_bond":
        return "fixed_income_core"
    if asset_class == "bond":
        return "fixed_income"
    if asset_class == "pension":
        return "long_term_retirement"
    if asset_class == "equity" and subtype == "broad_index":
        return "core_equity"
    if asset_class == "equity" and subtype == "dividend":
        return "defensive_equity"
    if asset_class == "equity" and subtype == "theme_growth":
        return "satellite_growth"
    return "general"


def volatility_to_risk_level(asset_class: str, volatility: float | None) -> int:
    if volatility is None or pd.isna(volatility):
        return {"cash": 1, "bond": 2, "pension": 3, "equity": 4}.get(asset_class, 3)

    if asset_class == "cash":
        return 1
    if asset_class == "bond":
        if volatility < 0.03:
            return 1
        if volatility < 0.07:
            return 2
        return 3
    if asset_class == "pension":
        if volatility < 0.08:
            return 2
        if volatility < 0.15:
            return 3
        return 4
    if asset_class == "equity":
        if volatility < 0.12:
            return 3
        if volatility < 0.25:
            return 4
        return 5
    return 3


def get_close_series(history: pd.DataFrame) -> pd.Series:
    if history is None or history.empty or "종가" not in history:
        return pd.Series(dtype=float)
    return pd.to_numeric(history["종가"], errors="coerce").dropna()


def compute_annualized_volatility(history: pd.DataFrame, min_returns: int) -> float | None:
    close = get_close_series(history)
    returns = close.pct_change().dropna()
    if len(returns) < min_returns:
        return None
    return float(returns.std() * np.sqrt(252))


def compute_return(close: pd.Series, periods: int | None = None) -> float | None:
    if close.empty or len(close) < 2:
        return None
    start = close.iloc[0] if periods is None or len(close) <= periods else close.iloc[-periods - 1]
    end = close.iloc[-1]
    if not start:
        return None
    return float(end / start - 1.0)


def compute_liquidity_metrics(history: pd.DataFrame) -> tuple[float | None, float | None, float | None]:
    if history is None or history.empty:
        return None, None, None

    if "거래대금" in history:
        liquidity = pd.to_numeric(history["거래대금"], errors="coerce").dropna()
    elif "거래량" in history:
        liquidity = pd.to_numeric(history["거래량"], errors="coerce").dropna()
    else:
        return None, None, None

    if liquidity.empty:
        return None, None, None

    average_20d = float(liquidity.tail(20).mean())
    average_60d = float(liquidity.tail(60).mean())
    latest = float(liquidity.iloc[-1])
    return average_20d, average_60d, latest


def percentile_score(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rank(method="average", pct=True).fillna(0.5)


def build_score(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["liquidity_score"] = percentile_score(result["raw_liquidity"])
    result["return_score"] = percentile_score(result["return_1y"])
    result["cost_score"] = 0.5
    result["role_score"] = result["subtype"].map(ROLE_SCORE).fillna(ROLE_SCORE["other"])

    penalty = pd.Series(0.0, index=result.index)
    penalty += result["exclude_reason"].notna().astype(float) * 1.0
    penalty += (result["asset_class"] == "other").astype(float) * 0.35
    penalty += result["volatility"].isna().astype(float) * 0.1

    result["score"] = (
        0.45 * result["liquidity_score"]
        + 0.2 * result["return_score"]
        + 0.15 * result["cost_score"]
        + 0.2 * result["role_score"]
        - penalty
    ).round(6)
    return result


def select_representative_rows(frame: pd.DataFrame, per_asset_class: int | None) -> pd.DataFrame:
    if per_asset_class is None or per_asset_class <= 0:
        return frame.sort_values(
            ["asset_class", "score", "raw_liquidity", "product_name", "product_id"],
            ascending=[True, False, False, True, True],
        ).reset_index(drop=True)

    selected_frames = []
    for asset_class in ASSET_CLASSES:
        group = frame[frame["asset_class"] == asset_class].copy()
        if group.empty:
            continue
        group = group.sort_values(
            ["score", "raw_liquidity", "return_1y", "product_name", "product_id"],
            ascending=[False, False, False, True, True],
        )

        selected_indices = []
        for subtype in group["subtype"].drop_duplicates().tolist():
            candidates = group[(group["subtype"] == subtype) & (~group.index.isin(selected_indices))]
            if not candidates.empty:
                selected_indices.append(candidates.index[0])
            if len(selected_indices) >= per_asset_class:
                break

        remaining = group.loc[~group.index.isin(selected_indices)]
        selected_indices.extend(remaining.head(per_asset_class - len(selected_indices)).index.tolist())
        selected_frames.append(group.loc[selected_indices])

    if not selected_frames:
        return frame.iloc[0:0].copy()

    return pd.concat(selected_frames, ignore_index=True).sort_values(
        ["asset_class", "score", "raw_liquidity", "product_name", "product_id"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)


def fetch_etf_history(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    from pykrx import stock

    return stock.get_etf_ohlcv_by_date(start_date, end_date, ticker)


def build_etf_product_pool(
    *,
    asset_classes: Iterable[str],
    risk_levels: Iterable[int] | None,
    min_returns: int,
    sleep_seconds: float,
    lookback_days: int,
    latest_lookback_days: int,
    max_tickers: int | None,
    per_asset_class: int | None,
    include_excluded: bool,
) -> pd.DataFrame:
    from pykrx import stock

    end_date, all_tickers = resolve_latest_trading_date(latest_lookback_days)
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=lookback_days)).strftime(
        "%Y%m%d"
    )

    if max_tickers is not None:
        all_tickers = all_tickers[:max_tickers]

    rows = []
    for index, ticker in enumerate(all_tickers, start=1):
        try:
            name = scalar_text(stock.get_etf_ticker_name(ticker))
        except Exception as error:
            print(f"[{index}/{len(all_tickers)}] {ticker} - 이름 조회 실패: {error}")
            time.sleep(sleep_seconds)
            continue

        try:
            history = fetch_etf_history(ticker, start_date, end_date)
        except Exception as error:
            print(f"[{index}/{len(all_tickers)}] {name} - OHLCV 조회 실패: {error}")
            history = pd.DataFrame()

        close = get_close_series(history)
        volatility = compute_annualized_volatility(history, min_returns=min_returns)
        return_1y = compute_return(close)
        return_3m = compute_return(close, periods=63)
        average_20d, average_60d, latest_liquidity = compute_liquidity_metrics(history)

        asset_class = classify_asset_class(name)
        subtype = classify_subtype(name)
        risk_level = volatility_to_risk_level(asset_class, volatility)
        exclude_reason = get_exclude_reason(name)
        theme = infer_theme(name, asset_class, subtype)

        rows.append(
            {
                "product_id": ticker,
                "product_name": name,
                "provider": infer_provider(name),
                "asset_class": asset_class,
                "subtype": subtype,
                "risk_level": risk_level,
                "expected_role": infer_expected_role(asset_class, subtype),
                "volatility": volatility,
                "return_1y": return_1y,
                "return_3m": return_3m,
                "raw_liquidity": average_60d,
                "avg_trading_value_20d": average_20d,
                "latest_trading_value": latest_liquidity,
                "exclude_reason": exclude_reason,
                "as_of_date": end_date,
                "market": infer_market(name),
                "theme": theme,
                "theme_risk_level": infer_theme_risk_level(name),
            }
        )
        vol_text = "N/A" if volatility is None else f"{volatility * 100:.2f}%"
        print(
            f"[{index}/{len(all_tickers)}] {name} - {asset_class}/{subtype}, "
            f"risk {risk_level}, vol {vol_text}"
        )
        time.sleep(sleep_seconds)

    if not rows:
        raise ValueError("ETF 상품 pool 생성 결과가 비어 있습니다.")

    frame = build_score(pd.DataFrame(rows))

    requested_asset_classes = set(asset_classes)
    frame = frame[frame["asset_class"].isin(requested_asset_classes)].copy()
    if risk_levels is not None:
        frame = frame[frame["risk_level"].isin(set(risk_levels))].copy()
    if not include_excluded:
        frame = frame[frame["exclude_reason"].isna()].copy()
    if frame.empty:
        raise ValueError("필터 조건에 맞는 ETF 상품이 없습니다.")

    selected = select_representative_rows(frame, per_asset_class)
    return with_legacy_columns(selected)


def with_legacy_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    volatility_percent = result["volatility"].mul(100.0)

    result["Ticker"] = result["product_id"]
    result["Name"] = result["product_name"]
    result["Volatility(%)"] = volatility_percent.round(2)
    result["Risk_Level"] = result["risk_level"]
    result["Theme_Risk_Level"] = result["theme_risk_level"]
    result["Provider"] = result["provider"]
    result["Market"] = result["market"]
    result["Theme"] = result["theme"]
    result["AvgTradingValue(20D)"] = result["avg_trading_value_20d"].round(2)
    result["LatestTradingValue"] = result["latest_trading_value"].round(2)
    result["Preference_Score"] = result["score"].round(6)
    result["Selection_Reason"] = result["expected_role"]
    result["AsOfDate"] = result["as_of_date"]
    result["Liquidity_Score"] = result["liquidity_score"].round(6)
    result["Stability_Score"] = (1.0 - percentile_score(result["volatility"])).round(6)
    result["Selection_Score"] = result["score"].round(6)

    ordered_columns = [
        "product_id",
        "product_name",
        "provider",
        "asset_class",
        "subtype",
        "risk_level",
        "expected_role",
        "volatility",
        "return_1y",
        "return_3m",
        "liquidity_score",
        "cost_score",
        "score",
        "exclude_reason",
        "raw_liquidity",
        "avg_trading_value_20d",
        "latest_trading_value",
        "as_of_date",
        "market",
        "theme",
        "theme_risk_level",
        "Ticker",
        "Name",
        "Volatility(%)",
        "Risk_Level",
        "Theme_Risk_Level",
        "Provider",
        "Market",
        "Theme",
        "AvgTradingValue(20D)",
        "LatestTradingValue",
        "Preference_Score",
        "Selection_Reason",
        "AsOfDate",
        "Liquidity_Score",
        "Stability_Score",
        "Selection_Score",
    ]
    return result[ordered_columns].reset_index(drop=True)


def parse_optional_int_list(values: list[int] | None) -> list[int] | None:
    if not values:
        return None
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "dataset" / "catalogs" / "pykrx_etf_snapshot.csv",
        help="ETF 추천 후보 CSV 저장 경로",
    )
    parser.add_argument(
        "--asset-classes",
        nargs="+",
        default=ASSET_CLASSES,
        choices=ASSET_CLASSES,
        help="CSV에 포함할 asset class 목록",
    )
    parser.add_argument(
        "--risk-levels",
        type=int,
        nargs="+",
        default=None,
        help="선택적으로 CSV에 포함할 risk level 목록",
    )
    parser.add_argument(
        "--per-asset-class",
        type=int,
        default=40,
        help="asset class별 대표 ETF 개수. 0 이하이면 제한하지 않음",
    )
    parser.add_argument(
        "--min-returns",
        type=int,
        default=30,
        help="변동성 계산에 필요한 최소 일별 수익률 개수",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="변동성/수익률 계산용 과거 조회 일수",
    )
    parser.add_argument(
        "--latest-lookback-days",
        type=int,
        default=10,
        help="최신 ETF 거래일 탐색 일수",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.5,
        help="ETF 조회 사이 대기 시간",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="테스트용 ETF 조회 상한",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="레버리지/인버스/합성 등 기본 제외 상품도 CSV에 포함",
    )
    args = parser.parse_args()

    snapshot = build_etf_product_pool(
        asset_classes=args.asset_classes,
        risk_levels=parse_optional_int_list(args.risk_levels),
        min_returns=args.min_returns,
        sleep_seconds=args.sleep_seconds,
        lookback_days=args.lookback_days,
        latest_lookback_days=args.latest_lookback_days,
        max_tickers=args.max_tickers,
        per_asset_class=args.per_asset_class,
        include_excluded=args.include_excluded,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(args.output_path, index=False, encoding="utf-8-sig")

    print()
    print(f"저장 완료: {args.output_path}")
    print(f"총 {len(snapshot)}개 ETF 저장")
    print(snapshot["asset_class"].value_counts().reindex(ASSET_CLASSES, fill_value=0).to_string())
    print()
    print(snapshot.groupby(["asset_class", "subtype"]).size().to_string())


if __name__ == "__main__":
    main()
