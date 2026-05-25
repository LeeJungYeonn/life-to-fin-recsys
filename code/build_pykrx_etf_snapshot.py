from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


THEME_RISK_RULES = {
    5: ["레버리지", "인버스", "2X", "바이오", "2차전지"],
    4: ["200", "S&P500", "나스닥100"],
    3: ["고배당", "배당성장", "커버드콜"],
    2: ["국고채", "단기채", "통안채", "특수채", "지방채", "달러단기", "AA", "AAA"],
}

THEME_KEYWORDS = {
    "bond": ["국고채", "단기채", "통안채", "특수채", "지방채", "회사채", "채권", "bond"],
    "cash": ["파킹", "머니마켓", "mmf"],
    "retirement": ["연금", "은퇴", "target date", "tdf"],
    "dividend": ["고배당", "배당성장", "커버드콜", "dividend"],
    "leveraged": ["레버리지", "2x"],
    "inverse": ["인버스"],
    "bio": ["바이오"],
    "battery": ["2차전지"],
    "nasdaq100": ["나스닥100", "nasdaq100"],
    "sp500": ["s&p500"],
    "index200": ["코스피200", "200"],
}

GLOBAL_MARKET_KEYWORDS = [
    "미국",
    "중국",
    "일본",
    "유럽",
    "인도",
    "베트남",
    "글로벌",
    "world",
    "global",
    "msci",
    "s&p",
    "nasdaq",
    "nikkei",
    "euro",
    "hong kong",
    "taiwan",
    "brazil",
]


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


def classify_volatility_risk_level(volatility_percent: float) -> int:
    if volatility_percent < 5.0:
        return 1
    if volatility_percent < 10.0:
        return 2
    if volatility_percent < 15.0:
        return 3
    if volatility_percent < 25.0:
        return 4
    return 5


def infer_theme_risk_level(name: str) -> int | None:
    for level in sorted(THEME_RISK_RULES.keys(), reverse=True):
        if any(token in name for token in THEME_RISK_RULES[level]):
            return level
    return None


def infer_theme(name: str) -> str:
    lowered = name.lower()
    for theme, keywords in THEME_KEYWORDS.items():
        if any(keyword in name or keyword in lowered for keyword in keywords):
            return theme
    return "equity"


def infer_market(name: str) -> str:
    lowered = name.lower()
    if any(keyword in name or keyword in lowered for keyword in GLOBAL_MARKET_KEYWORDS):
        return "global"
    return "domestic"


def infer_provider(name: str) -> str:
    tokens = name.split()
    return tokens[0] if tokens else "ETF"


def select_representative_rows(frame: pd.DataFrame, per_risk_level: int) -> pd.DataFrame:
    selected_frames = []
    for risk_level in sorted(frame["Risk_Level"].unique()):
        group = frame[frame["Risk_Level"] == risk_level].sort_values(
            ["Volatility(%)", "Name", "Ticker"]
        )
        if len(group) <= per_risk_level:
            selected_frames.append(group)
            continue

        raw_indices = np.linspace(0, len(group) - 1, per_risk_level)
        indices = sorted({int(round(value)) for value in raw_indices})
        if len(indices) < per_risk_level:
            for candidate in range(len(group)):
                if candidate not in indices:
                    indices.append(candidate)
                if len(indices) == per_risk_level:
                    break
            indices.sort()
        selected_frames.append(group.iloc[indices])

    result = pd.concat(selected_frames, ignore_index=True)
    return result.sort_values(["Risk_Level", "Volatility(%)", "Name", "Ticker"]).reset_index(
        drop=True
    )


def build_snapshot(
    *,
    risk_levels: Iterable[int],
    per_risk_level: int,
    min_history: int,
    sleep_seconds: float,
    lookback_days: int,
    max_tickers: int | None,
) -> pd.DataFrame:
    from pykrx import stock

    end_date, all_tickers = resolve_latest_trading_date()
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=lookback_days)).strftime(
        "%Y%m%d"
    )

    if max_tickers is not None:
        all_tickers = all_tickers[:max_tickers]

    rows = []
    for index, ticker in enumerate(all_tickers, start=1):
        name = stock.get_etf_ticker_name(ticker)
        try:
            history = stock.get_etf_ohlcv_by_date(start_date, end_date, ticker)
        except Exception as error:
            print(f"[{index}/{len(all_tickers)}] {name} - OHLCV 조회 실패: {error}")
            time.sleep(sleep_seconds)
            continue

        if len(history) < min_history:
            print(f"[{index}/{len(all_tickers)}] {name} - 상장 기간 부족으로 제외")
            time.sleep(sleep_seconds)
            continue

        daily_returns = history["종가"].pct_change().dropna()
        if daily_returns.empty:
            print(f"[{index}/{len(all_tickers)}] {name} - 수익률 계산 불가")
            time.sleep(sleep_seconds)
            continue

        volatility_percent = float(daily_returns.std() * np.sqrt(252) * 100.0)
        rows.append(
            {
                "Ticker": ticker,
                "Name": name,
                "Volatility(%)": round(volatility_percent, 2),
                "Risk_Level": classify_volatility_risk_level(volatility_percent),
                "Theme_Risk_Level": infer_theme_risk_level(name),
                "Provider": infer_provider(name),
                "Market": infer_market(name),
                "Theme": infer_theme(name),
                "AsOfDate": end_date,
            }
        )
        print(f"[{index}/{len(all_tickers)}] {name} - 변동성 {volatility_percent:.2f}%")
        time.sleep(sleep_seconds)

    if not rows:
        raise ValueError("ETF 스냅샷 생성 결과가 비어 있습니다.")

    frame = pd.DataFrame(rows).sort_values(["Risk_Level", "Volatility(%)", "Name", "Ticker"])
    frame = frame[frame["Risk_Level"].isin(list(risk_levels))].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"선택한 risk_level={list(risk_levels)} 에 해당하는 ETF가 없습니다.")
    return select_representative_rows(frame, per_risk_level)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("dataset/catalogs/pykrx_etf_snapshot.csv"),
        help="ETF 추천 후보 CSV 저장 경로",
    )
    parser.add_argument(
        "--risk-levels",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5],
        help="CSV에 포함할 ETF risk level 목록",
    )
    parser.add_argument(
        "--per-risk-level",
        type=int,
        default=8,
        help="risk level별 대표 ETF 개수",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=120,
        help="변동성 계산에 필요한 최소 거래일 수",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="변동성 계산용 과거 조회 일수",
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
    args = parser.parse_args()

    snapshot = build_snapshot(
        risk_levels=args.risk_levels,
        per_risk_level=args.per_risk_level,
        min_history=args.min_history,
        sleep_seconds=args.sleep_seconds,
        lookback_days=args.lookback_days,
        max_tickers=args.max_tickers,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(args.output_path, index=False, encoding="utf-8-sig")

    print()
    print(f"저장 완료: {args.output_path}")
    print(f"총 {len(snapshot)}개 ETF 저장")
    print(snapshot["Risk_Level"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
