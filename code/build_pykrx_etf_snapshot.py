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
    "quality": ["퀄리티", "quality", "우량", "블루칩"],
    "leveraged": ["레버리지", "2x"],
    "inverse": ["인버스"],
    "bio": ["바이오"],
    "battery": ["2차전지"],
    "semiconductor": ["반도체", "semiconductor"],
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

QUALITY_THEME_SCORES = {
    2: {
        "bond": 6.0,
        "cash": 5.5,
        "quality": 4.5,
        "dividend": 4.0,
        "sp500": 3.5,
        "index200": 3.0,
        "retirement": 3.0,
        "nasdaq100": 2.0,
        "equity": 1.0,
    },
    3: {
        "quality": 5.0,
        "dividend": 4.5,
        "sp500": 4.5,
        "retirement": 4.0,
        "index200": 4.0,
        "nasdaq100": 3.5,
        "bond": 3.0,
        "semiconductor": 3.0,
        "equity": 2.0,
    },
    4: {
        "quality": 5.5,
        "sp500": 5.5,
        "nasdaq100": 5.0,
        "semiconductor": 4.5,
        "index200": 4.0,
        "dividend": 3.5,
        "equity": 2.0,
        "bond": 1.0,
    },
    5: {
        "quality": 5.5,
        "nasdaq100": 5.5,
        "semiconductor": 5.0,
        "sp500": 4.5,
        "dividend": 2.5,
        "index200": 2.5,
        "equity": 2.0,
    },
}

CORE_QUALITY_KEYWORDS = [
    "s&p500",
    "나스닥100",
    "nasdaq100",
    "코스피200",
    "msci",
    "배당귀족",
    "퀄리티",
    "quality",
    "우량",
    "블루칩",
]

PROMISING_GROWTH_KEYWORDS = [
    "반도체",
    "semiconductor",
    "ai테크",
    "인공지능",
    "전력",
    "인프라",
]

SPECULATIVE_THEME_KEYWORDS = [
    "레버리지",
    "인버스",
    "2x",
    "바이오",
    "2차전지",
    "메타버스",
    "게임",
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


def compute_preference_score(
    name: str,
    *,
    risk_level: int,
    theme: str,
    theme_risk_level: int | None,
) -> tuple[float, str]:
    lowered = name.lower()
    score = QUALITY_THEME_SCORES.get(risk_level, {}).get(theme, 0.0)
    reasons = []

    if any(keyword in lowered for keyword in CORE_QUALITY_KEYWORDS):
        score += 3.0
        reasons.append("core_index")

    if any(keyword in lowered for keyword in PROMISING_GROWTH_KEYWORDS):
        score += 2.0 if risk_level >= 4 else 1.0
        reasons.append("growth_sector")

    if any(keyword in lowered for keyword in SPECULATIVE_THEME_KEYWORDS):
        score -= 6.0
        reasons.append("speculative_penalty")

    if theme_risk_level is not None and theme_risk_level >= 5:
        score -= 1.5
        reasons.append("theme_risk_penalty")

    return score, ",".join(reasons) if reasons else "balanced"


def compute_liquidity_metrics(history: pd.DataFrame) -> tuple[float, float]:
    if "거래대금" not in history:
        return 0.0, 0.0

    trading_value = history["거래대금"].dropna()
    if trading_value.empty:
        return 0.0, 0.0

    average_20d = float(trading_value.tail(20).mean())
    latest = float(trading_value.iloc[-1])
    return average_20d, latest


def preferred_themes_for_risk_level(risk_level: int) -> list[str]:
    theme_scores = QUALITY_THEME_SCORES.get(risk_level, {})
    return [
        theme
        for theme, _ in sorted(theme_scores.items(), key=lambda item: (-item[1], item[0]))
        if theme != "equity"
    ]


def select_representative_rows(frame: pd.DataFrame, per_risk_level: int) -> pd.DataFrame:
    selected_frames = []
    for risk_level in sorted(frame["Risk_Level"].unique()):
        group = frame[frame["Risk_Level"] == risk_level].copy()
        group["Liquidity_Score"] = group["AvgTradingValue(20D)"].rank(
            method="average", pct=True
        ).fillna(0.0) * 4.0
        group["Stability_Score"] = (
            1.0 - group["Volatility(%)"].rank(method="average", pct=True).fillna(1.0)
        ) * 1.5
        group["Selection_Score"] = (
            group["Preference_Score"] + group["Liquidity_Score"] + group["Stability_Score"]
        )
        group = group.sort_values(
            ["Selection_Score", "AvgTradingValue(20D)", "Volatility(%)", "Name", "Ticker"],
            ascending=[False, False, True, True, True],
        )
        if len(group) <= per_risk_level:
            selected_frames.append(group)
            continue

        selected_indices = []
        selected_themes = set()

        for theme in preferred_themes_for_risk_level(int(risk_level)):
            candidates = group[
                (group["Theme"] == theme) & (~group.index.isin(selected_indices))
            ]
            if candidates.empty:
                continue
            selected_indices.append(candidates.index[0])
            selected_themes.add(theme)
            if len(selected_indices) == per_risk_level:
                break

        remaining = group.loc[~group.index.isin(selected_indices)].copy()
        remaining["ThemeSeen"] = remaining["Theme"].isin(selected_themes)
        remaining = remaining.sort_values(
            ["ThemeSeen", "Selection_Score", "AvgTradingValue(20D)", "Volatility(%)", "Name", "Ticker"],
            ascending=[True, False, False, True, True, True],
        )
        selected_indices.extend(remaining.head(per_risk_level - len(selected_indices)).index.tolist())
        selected_frames.append(group.loc[selected_indices])

    result = pd.concat(selected_frames, ignore_index=True)
    if "ThemeSeen" in result.columns:
        result = result.drop(columns=["ThemeSeen"])
    return result.sort_values(
        ["Risk_Level", "Selection_Score", "AvgTradingValue(20D)", "Volatility(%)", "Name", "Ticker"],
        ascending=[True, False, False, True, True, True],
    ).reset_index(drop=True)


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
        risk_level = classify_volatility_risk_level(volatility_percent)
        theme = infer_theme(name)
        theme_risk_level = infer_theme_risk_level(name)
        average_trading_value, latest_trading_value = compute_liquidity_metrics(history)
        preference_score, selection_reason = compute_preference_score(
            name,
            risk_level=risk_level,
            theme=theme,
            theme_risk_level=theme_risk_level,
        )
        rows.append(
            {
                "Ticker": ticker,
                "Name": name,
                "Volatility(%)": round(volatility_percent, 2),
                "Risk_Level": risk_level,
                "Theme_Risk_Level": theme_risk_level,
                "Provider": infer_provider(name),
                "Market": infer_market(name),
                "Theme": theme,
                "AvgTradingValue(20D)": round(average_trading_value, 2),
                "LatestTradingValue": round(latest_trading_value, 2),
                "Preference_Score": round(preference_score, 2),
                "Selection_Reason": selection_reason,
                "AsOfDate": end_date,
            }
        )
        print(f"[{index}/{len(all_tickers)}] {name} - 변동성 {volatility_percent:.2f}%")
        time.sleep(sleep_seconds)

    if not rows:
        raise ValueError("ETF 스냅샷 생성 결과가 비어 있습니다.")

    frame = pd.DataFrame(rows).sort_values(
        ["Risk_Level", "Preference_Score", "AvgTradingValue(20D)", "Volatility(%)", "Name", "Ticker"],
        ascending=[True, False, False, True, True, True],
    )
    frame = frame[frame["Risk_Level"].isin(list(risk_levels))].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"선택한 risk_level={list(risk_levels)} 에 해당하는 ETF가 없습니다.")
    return select_representative_rows(frame, per_risk_level)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "dataset" / "catalogs" / "pykrx_etf_snapshot.csv",
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
