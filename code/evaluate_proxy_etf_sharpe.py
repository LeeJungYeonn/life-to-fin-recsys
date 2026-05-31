from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from evaluate_baselines import (
    CatBoostAllocationBaseline,
    GroupMeanAllocationBaseline,
    MeanAllocationBaseline,
    _frame_to_markdown,
    _load_split,
    _normalize_alloc,
    source_encoder_predictions,
)
from portfolio_schema import BUCKET_COLUMNS, CATEGORICAL_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_ETF = {
    "ticker": "069500",
    "name": "KODEX 200",
}

PROXY_ETFS = {
    "cash": {
        "ticker": "423160",
        "name": "KODEX KOFR금리액티브(합성)",
    },
    "bond": {
        "ticker": "114260",
        "name": "KODEX 국고채3년",
    },
    "pension": {
        "ticker": "433970",
        "name": "KODEX TDF2030액티브 적격",
    },
    "equity": {
        "ticker": "069500",
        "name": "KODEX 200",
    },
}

DEFAULT_SOURCE_PREFIXES = (
    "supcon_final=allocation_best"
)
DEFAULT_MODELS = "supcon_final"
DEFAULT_RISK_LABELS = "1,2,3,4,5"


def load_krx_env() -> None:
    env_paths = [
        REPO_ROOT.parent / ".env",
        REPO_ROOT / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=False)


def _date_arg(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _date_display(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _parse_source_prefixes(value: str) -> list[tuple[str, str]]:
    pairs = []
    for item in value.split(","):
        if not item.strip():
            continue
        label, prefix = item.split("=", maxsplit=1)
        pairs.append((label.strip(), prefix.strip()))
    return pairs


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def make_random_periods(
    *,
    start_date: str,
    end_date: str,
    num_periods: int,
    min_days: int,
    max_days: int,
    seed: int,
) -> list[tuple[int, str, str]]:
    if num_periods <= 0:
        return [(1, start_date, end_date)]
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    total_days = int((end_ts - start_ts).days)
    if total_days < min_days:
        raise ValueError("Random period range is shorter than --random-period-min-days.")
    max_days = min(max_days, total_days)
    if min_days > max_days:
        raise ValueError("--random-period-min-days cannot be greater than --random-period-max-days.")

    rng = np.random.default_rng(seed)
    periods = []
    for period_id in range(1, num_periods + 1):
        length = int(rng.integers(min_days, max_days + 1))
        latest_start_offset = total_days - length
        offset = int(rng.integers(0, latest_start_offset + 1))
        period_start = start_ts + pd.Timedelta(days=offset)
        period_end = period_start + pd.Timedelta(days=length)
        periods.append((period_id, period_start.strftime("%Y%m%d"), period_end.strftime("%Y%m%d")))
    return periods


def sample_indices_by_risk_label(
    *,
    test_csv_path: Path,
    risk_label_col: str,
    risk_labels: list[int],
    samples_per_label: int,
    seed: int,
    expected_rows: int,
) -> dict[int, np.ndarray]:
    test_frame = pd.read_csv(test_csv_path)
    if len(test_frame) != expected_rows:
        raise ValueError(
            f"{test_csv_path} has {len(test_frame)} rows, but processed test tensors have {expected_rows} rows."
        )
    if risk_label_col not in test_frame.columns:
        raise ValueError(f"Column '{risk_label_col}' is missing from {test_csv_path}.")

    rng = np.random.default_rng(seed)
    indices_by_label: dict[int, np.ndarray] = {}
    for label in risk_labels:
        label_indices = test_frame.index[test_frame[risk_label_col] == label].to_numpy(dtype=np.int64)
        if len(label_indices) < samples_per_label:
            raise ValueError(
                f"RISK_LABEL={label} has only {len(label_indices)} rows; "
                f"cannot sample {samples_per_label}."
            )
        sampled = rng.choice(label_indices, size=samples_per_label, replace=False)
        indices_by_label[label] = np.sort(sampled)
    return indices_by_label


def load_batch_allocations_for_indices(
    *,
    batch_details_path: Path,
    test_csv_path: Path,
    test_indices: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    test_frame = pd.read_csv(test_csv_path)
    details = json.loads(batch_details_path.read_text(encoding="utf-8"))
    rows = details.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{batch_details_path} does not contain a list at key 'rows'.")

    row_by_caseid = {int(row["CASEID"]): row for row in rows}
    caseids = [int(test_frame.iloc[int(idx)]["CASEID"]) for idx in test_indices]
    allocations = []
    missing_caseids = []
    for caseid in caseids:
        row = row_by_caseid.get(caseid)
        if row is None:
            missing_caseids.append(caseid)
            continue
        allocations.append([float(row[f"pred_{bucket}"]) for bucket in BUCKET_COLUMNS])

    if missing_caseids:
        raise KeyError(
            f"{len(missing_caseids)} sampled CASEID values were not found in {batch_details_path}: "
            + ", ".join(str(caseid) for caseid in missing_caseids[:10])
        )
    return _normalize_alloc(np.asarray(allocations, dtype=np.float32)), caseids


def _checkpoint_dir_for_source_label(
    source_label: str,
    checkpoint_dir: Path,
    source_tuned_checkpoint_dir: Path,
) -> Path:
    if source_label == "source_tuned":
        return source_tuned_checkpoint_dir
    return checkpoint_dir


def _load_frames(processed_dir: Path) -> tuple[torch.Tensor, np.ndarray, np.ndarray, torch.Tensor, pd.DataFrame, pd.DataFrame]:
    train_x_cat_t, train_alloc, train_risky, _ = _load_split(processed_dir, "train")
    test_x_cat_t, test_alloc, _, test_labels_t = _load_split(processed_dir, "test")

    train_x_cat = pd.DataFrame(train_x_cat_t.numpy(), columns=CATEGORICAL_COLUMNS).astype(str)
    test_x_cat = pd.DataFrame(test_x_cat_t.numpy(), columns=CATEGORICAL_COLUMNS).astype(str)
    return train_x_cat_t, train_alloc, train_risky, test_labels_t, train_x_cat, test_x_cat


def build_model_predictions(
    *,
    processed_dir: Path,
    checkpoint_dir: Path,
    source_tuned_checkpoint_dir: Path,
    source_prefixes: list[tuple[str, str]],
    model_names: Iterable[str],
    test_indices: np.ndarray | None,
    seed: int,
    catboost_iterations: int,
    group_min_count: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    train_x_cat_t, train_alloc, train_risky, test_labels_t, train_x_cat, test_x_cat = _load_frames(processed_dir)
    test_x_cat_t, test_alloc, _, _ = _load_split(processed_dir, "test")
    if test_indices is not None:
        test_x_cat_t = test_x_cat_t[test_indices]
        test_alloc = test_alloc[test_indices]
        test_labels_t = test_labels_t[test_indices]
        test_x_cat = test_x_cat.iloc[test_indices].reset_index(drop=True)

    predictions: dict[str, np.ndarray] = {}
    requested = set(model_names)

    if "mean_allocation" in requested:
        mean_model = MeanAllocationBaseline().fit(train_alloc, train_risky)
        pred_alloc, _ = mean_model.predict(test_x_cat)
        predictions["mean_allocation"] = pred_alloc

    if "group_mean_AGECL_HOUSECL_EDCL_OCCAT1" in requested:
        group_cols = ["AGECL", "HOUSECL", "EDCL", "OCCAT1"]
        group_model = GroupMeanAllocationBaseline(group_cols, min_count=group_min_count).fit(
            train_x_cat,
            train_alloc,
            train_risky,
        )
        pred_alloc, _ = group_model.predict(test_x_cat)
        predictions["group_mean_AGECL_HOUSECL_EDCL_OCCAT1"] = pred_alloc

    if "catboost" in requested:
        train_idx, valid_idx = train_test_split(
            np.arange(len(train_x_cat)),
            test_size=0.15,
            random_state=seed,
            shuffle=True,
        )
        catboost_model = CatBoostAllocationBaseline(
            cat_features=list(range(len(CATEGORICAL_COLUMNS))),
            iterations=catboost_iterations,
            seed=seed,
        ).fit(
            train_x_cat.iloc[train_idx],
            train_alloc[train_idx],
            train_risky[train_idx],
            train_x_cat.iloc[valid_idx],
            train_alloc[valid_idx],
            train_risky[valid_idx],
        )
        pred_alloc, _ = catboost_model.predict(test_x_cat)
        predictions["catboost"] = pred_alloc

    for source_label, source_prefix in source_prefixes:
        if source_label not in requested:
            continue
        source_checkpoint_dir = _checkpoint_dir_for_source_label(
            source_label,
            checkpoint_dir,
            source_tuned_checkpoint_dir,
        )
        source_alloc, _ = source_encoder_predictions(
            source_checkpoint_dir,
            source_prefix,
            processed_dir,
            test_x_cat_t,
            test_alloc,
            test_labels_t,
            device,
        )
        predictions[source_label] = source_alloc

    return {name: _normalize_alloc(alloc) for name, alloc in predictions.items()}


def fetch_pykrx_price_frame(
    *,
    start_date: str,
    end_date: str,
    price_column: str,
    fill_method: str,
) -> pd.DataFrame:
    from pykrx import stock

    series: dict[str, pd.Series] = {}
    etfs = {bucket: payload for bucket, payload in PROXY_ETFS.items()}
    etfs["benchmark"] = BENCHMARK_ETF

    for label, payload in etfs.items():
        frame = stock.get_etf_ohlcv_by_date(start_date, end_date, payload["ticker"])
        if frame.empty:
            raise ValueError(f"PyKrx returned no rows for {payload['name']} ({payload['ticker']}).")

        if price_column not in frame.columns:
            available = ", ".join(str(col) for col in frame.columns)
            raise ValueError(
                f"Column '{price_column}' is missing for {payload['name']} ({payload['ticker']}). "
                f"Available columns: {available}"
            )

        values = pd.to_numeric(frame[price_column], errors="coerce")
        values = values.mask(values <= 0.0)
        if fill_method == "ffill":
            values = values.ffill()
        series[label] = values.rename(label)

    prices = pd.concat(series.values(), axis=1).sort_index()
    prices.index = pd.to_datetime(prices.index)
    return prices


def _annualized_stats(
    returns: np.ndarray,
    *,
    annualization: int,
    risk_free_rate: float,
) -> dict[str, float]:
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return {
            "annualized_return": float("nan"),
            "annualized_volatility": float("nan"),
            "sharpe_ratio": float("nan"),
            "cumulative_return": float("nan"),
        }

    excess = returns - (risk_free_rate / annualization)
    mean_excess = float(np.mean(excess))
    daily_vol = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    sharpe = np.nan if daily_vol <= 0.0 else mean_excess / daily_vol * np.sqrt(annualization)
    cumulative = float(np.prod(1.0 + returns) - 1.0)
    annualized_return = float((1.0 + cumulative) ** (annualization / returns.size) - 1.0)
    annualized_vol = float(daily_vol * np.sqrt(annualization))
    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": float(sharpe),
        "cumulative_return": cumulative,
    }


def _finite_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def evaluate_proxy_sharpe(
    model_name: str,
    allocations: np.ndarray,
    proxy_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    *,
    annualization: int,
    risk_free_rate: float,
) -> dict[str, object]:
    allocations = _normalize_alloc(allocations)
    benchmark_stats = _annualized_stats(
        benchmark_returns.to_numpy(),
        annualization=annualization,
        risk_free_rate=risk_free_rate,
    )

    aggregate_allocation = allocations.mean(axis=0)
    aggregate_returns = proxy_returns[BUCKET_COLUMNS].to_numpy() @ aggregate_allocation
    aggregate_stats = _annualized_stats(
        aggregate_returns,
        annualization=annualization,
        risk_free_rate=risk_free_rate,
    )

    user_returns = proxy_returns[BUCKET_COLUMNS].to_numpy() @ allocations.T
    user_sharpes = np.array(
        [
            _annualized_stats(
                user_returns[:, idx],
                annualization=annualization,
                risk_free_rate=risk_free_rate,
            )["sharpe_ratio"]
            for idx in range(user_returns.shape[1])
        ],
        dtype=np.float64,
    )

    benchmark_sharpe = float(benchmark_stats["sharpe_ratio"])
    hit_rate = float(np.mean(user_sharpes > benchmark_sharpe)) if np.isfinite(benchmark_sharpe) else float("nan")

    row: dict[str, object] = {
        "model": model_name,
        "num_test_users": int(allocations.shape[0]),
        "num_return_days": int(len(proxy_returns)),
        "benchmark_sharpe": benchmark_sharpe,
        "aggregate_sharpe": aggregate_stats["sharpe_ratio"],
        "aggregate_excess_sharpe": float(aggregate_stats["sharpe_ratio"] - benchmark_sharpe),
        "user_sharpe_hit_rate": hit_rate,
        "aggregate_annualized_return": aggregate_stats["annualized_return"],
        "aggregate_annualized_volatility": aggregate_stats["annualized_volatility"],
        "aggregate_cumulative_return": aggregate_stats["cumulative_return"],
        "benchmark_annualized_return": benchmark_stats["annualized_return"],
        "benchmark_annualized_volatility": benchmark_stats["annualized_volatility"],
        "benchmark_cumulative_return": benchmark_stats["cumulative_return"],
    }
    row.update(_finite_summary(user_sharpes, "user_sharpe"))
    for bucket, weight in zip(BUCKET_COLUMNS, aggregate_allocation):
        row[f"mean_weight_{bucket}"] = float(weight)
    return row


def build_comparison_rows(
    proxy_rows: list[dict[str, object]],
    *,
    group: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    if not proxy_rows:
        return []

    group = group or {}
    rows: list[dict[str, object]] = []
    for proxy_row in proxy_rows:
        row = {
            "portfolio": f"{proxy_row['model']}_proxy_etf",
            "sharpe_ratio": proxy_row["aggregate_sharpe"],
            "excess_sharpe_vs_benchmark": proxy_row["aggregate_excess_sharpe"],
            "annualized_return": proxy_row["aggregate_annualized_return"],
            "annualized_volatility": proxy_row["aggregate_annualized_volatility"],
            "cumulative_return": proxy_row["aggregate_cumulative_return"],
            "num_return_days": proxy_row["num_return_days"],
            "num_test_users": proxy_row["num_test_users"],
            "user_sharpe_hit_rate_vs_benchmark": proxy_row["user_sharpe_hit_rate"],
            "user_sharpe_mean": proxy_row["user_sharpe_mean"],
            "user_sharpe_median": proxy_row["user_sharpe_median"],
            "mean_weight_cash": proxy_row["mean_weight_cash"],
            "mean_weight_bond": proxy_row["mean_weight_bond"],
            "mean_weight_pension": proxy_row["mean_weight_pension"],
            "mean_weight_equity": proxy_row["mean_weight_equity"],
        }
        rows.append({**group, **row})

    benchmark = proxy_rows[0]
    rows.append({
        **group,
        "portfolio": "benchmark_etf",
        "sharpe_ratio": benchmark["benchmark_sharpe"],
        "excess_sharpe_vs_benchmark": 0.0,
        "annualized_return": benchmark["benchmark_annualized_return"],
        "annualized_volatility": benchmark["benchmark_annualized_volatility"],
        "cumulative_return": benchmark["benchmark_cumulative_return"],
        "num_return_days": benchmark["num_return_days"],
        "num_test_users": np.nan,
        "user_sharpe_hit_rate_vs_benchmark": np.nan,
        "user_sharpe_mean": np.nan,
        "user_sharpe_median": np.nan,
        "mean_weight_cash": 0.0,
        "mean_weight_bond": 0.0,
        "mean_weight_pension": 0.0,
        "mean_weight_equity": 1.0,
    })
    return rows


def average_comparison_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []

    group_cols = ["risk_label", "portfolio"]
    mean_cols = [
        "sharpe_ratio",
        "excess_sharpe_vs_benchmark",
        "annualized_return",
        "annualized_volatility",
        "cumulative_return",
        "num_return_days",
        "num_test_users",
        "user_sharpe_hit_rate_vs_benchmark",
        "user_sharpe_mean",
        "user_sharpe_median",
        "mean_weight_cash",
        "mean_weight_bond",
        "mean_weight_pension",
        "mean_weight_equity",
    ]
    averaged = frame.groupby(group_cols, dropna=False)[mean_cols].mean(numeric_only=True).reset_index()
    period_counts = frame.groupby(group_cols, dropna=False)["period_id"].nunique().reset_index(name="num_periods")
    sample_sizes = frame.groupby(group_cols, dropna=False)["sample_size"].first().reset_index()
    averaged = averaged.merge(period_counts, on=group_cols, how="left")
    averaged = averaged.merge(sample_sizes, on=group_cols, how="left")
    columns = ["risk_label", "sample_size", "portfolio", "num_periods"] + [
        col for col in averaged.columns if col not in {"risk_label", "sample_size", "portfolio", "num_periods"}
    ]
    averaged = averaged[columns]
    return averaged.to_dict(orient="records")


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _select_models(predictions: dict[str, np.ndarray], names: Iterable[str]) -> dict[str, np.ndarray]:
    missing = [name for name in names if name not in predictions]
    if missing:
        raise KeyError(f"Missing model predictions: {', '.join(missing)}")
    return {name: predictions[name] for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "dataset" / "processed")
    parser.add_argument("--test-csv", type=Path, default=REPO_ROOT / "dataset" / "test.csv")
    parser.add_argument(
        "--batch-details",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "batch_end_to_end_details.json",
    )
    parser.add_argument("--risk-label-col", type=str, default="RISK_LABEL")
    parser.add_argument("--risk-labels", type=str, default=DEFAULT_RISK_LABELS)
    parser.add_argument("--samples-per-risk-label", type=int, default=20)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints",
        help="Directory for SupCon/allocation_best checkpoints.",
    )
    parser.add_argument(
        "--source-tuned-checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "grid_search",
        help="Directory for source_tuned checkpoints.",
    )
    parser.add_argument("--source-prefixes", type=str, default=DEFAULT_SOURCE_PREFIXES)
    parser.add_argument(
        "--models",
        type=str,
        default=DEFAULT_MODELS,
        help="Comma-separated model names to include in the proxy ETF comparison.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "checkpoints" / "proxy_etf_sharpe")
    parser.add_argument("--start-date", type=_date_arg, default="20220101")
    parser.add_argument("--end-date", type=_date_arg, default="20241231")
    parser.add_argument("--random-periods", type=int, default=0)
    parser.add_argument("--random-period-start-date", type=_date_arg, default="20200101")
    parser.add_argument("--random-period-end-date", type=_date_arg, default=date.today().strftime("%Y%m%d"))
    parser.add_argument("--random-period-min-days", type=int, default=365)
    parser.add_argument("--random-period-max-days", type=int, default=900)
    parser.add_argument("--period-seed", type=int, default=None)
    # parser.add_argument("--end-date", type=_date_arg, default=date.today().strftime("%Y%m%d"))
    parser.add_argument("--price-column", type=str, default="종가")
    parser.add_argument("--fill-method", choices=["none", "ffill"], default="ffill")
    parser.add_argument("--annualization", type=int, default=252)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--catboost-iterations", type=int, default=300)
    parser.add_argument("--group-min-count", type=int, default=30)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    load_krx_env()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    source_prefixes = _parse_source_prefixes(args.source_prefixes)
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    if len(model_names) != 1:
        raise ValueError("Batch-detail allocation evaluation expects exactly one model name.")
    risk_labels = _parse_int_list(args.risk_labels)

    expected_test_rows = int(torch.load(args.processed_dir / "test_x_cat_tensor.pt").shape[0])
    indices_by_label = sample_indices_by_risk_label(
        test_csv_path=args.test_csv,
        risk_label_col=args.risk_label_col,
        risk_labels=risk_labels,
        samples_per_label=args.samples_per_risk_label,
        seed=args.seed,
        expected_rows=expected_test_rows,
    )

    detail_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    sample_index_payload: dict[str, list[int]] = {}
    sample_caseid_payload: dict[str, list[int]] = {}
    period_seed = args.seed if args.period_seed is None else args.period_seed
    periods = make_random_periods(
        start_date=args.random_period_start_date if args.random_periods else args.start_date,
        end_date=args.random_period_end_date if args.random_periods else args.end_date,
        num_periods=args.random_periods,
        min_days=args.random_period_min_days,
        max_days=args.random_period_max_days,
        seed=period_seed,
    )
    period_payload: list[dict[str, object]] = []

    for period_id, period_start, period_end in periods:
        prices = fetch_pykrx_price_frame(
            start_date=period_start,
            end_date=period_end,
            price_column=args.price_column,
            fill_method=args.fill_method,
        )
        prices = prices.dropna(subset=BUCKET_COLUMNS + ["benchmark"], how="any")
        returns = prices.pct_change().dropna(how="any")
        if returns.empty:
            raise ValueError(
                f"No overlapping return rows remain for period {period_id}: "
                f"{period_start} to {period_end}."
            )
        period_payload.append(
            {
                "period_id": period_id,
                "requested_start": _date_display(period_start),
                "requested_end": _date_display(period_end),
                "price_start": returns.index.min(),
                "price_end": returns.index.max(),
                "num_return_days": int(len(returns)),
            }
        )

        for risk_label, test_indices in indices_by_label.items():
            allocations, caseids = load_batch_allocations_for_indices(
                batch_details_path=args.batch_details,
                test_csv_path=args.test_csv,
                test_indices=test_indices,
            )
            selected_predictions = {model_names[0]: allocations}

            rows = [
                evaluate_proxy_sharpe(
                    model_name,
                    allocation,
                    returns[BUCKET_COLUMNS],
                    returns["benchmark"],
                    annualization=args.annualization,
                    risk_free_rate=args.risk_free_rate,
                )
                for model_name, allocation in selected_predictions.items()
            ]
            group = {
                "period_id": period_id,
                "requested_start": _date_display(period_start),
                "requested_end": _date_display(period_end),
                "price_start": returns.index.min(),
                "price_end": returns.index.max(),
                "risk_label": risk_label,
                "sample_size": int(len(test_indices)),
            }
            comparison_rows.extend(build_comparison_rows(rows, group=group))
            for row in rows:
                detail_rows.append({**group, **row})
            sample_index_payload[str(risk_label)] = [int(idx) for idx in test_indices.tolist()]
            sample_caseid_payload[str(risk_label)] = caseids

    averaged_rows = average_comparison_rows(comparison_rows) if args.random_periods else comparison_rows
    frame = pd.DataFrame(averaged_rows)
    csv_path = args.output_dir / "proxy_etf_sharpe.csv"
    json_path = args.output_dir / "proxy_etf_sharpe.json"
    md_path = args.output_dir / "proxy_etf_sharpe.md"
    period_csv_path = args.output_dir / "proxy_etf_sharpe_periods.csv"

    frame.to_csv(csv_path, index=False)
    if args.random_periods:
        pd.DataFrame(comparison_rows).to_csv(period_csv_path, index=False)
    md_path.write_text(_frame_to_markdown(frame), encoding="utf-8")
    report = {
        "config": {
            "processed_dir": str(args.processed_dir),
            "test_csv": str(args.test_csv),
            "batch_details": str(args.batch_details),
            "allocation_source": "batch_end_to_end_details.pred_*",
            "risk_label_col": args.risk_label_col,
            "risk_labels": risk_labels,
            "samples_per_risk_label": args.samples_per_risk_label,
            "sample_indices_by_risk_label": sample_index_payload,
            "sample_caseids_by_risk_label": sample_caseid_payload,
            "checkpoint_dir": str(args.checkpoint_dir),
            "source_tuned_checkpoint_dir": str(args.source_tuned_checkpoint_dir),
            "source_prefixes": source_prefixes,
            "models": model_names,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "random_periods": args.random_periods,
            "random_period_start_date": args.random_period_start_date,
            "random_period_end_date": args.random_period_end_date,
            "random_period_min_days": args.random_period_min_days,
            "random_period_max_days": args.random_period_max_days,
            "period_seed": period_seed,
            "periods": period_payload,
            "price_column": args.price_column,
            "fill_method": args.fill_method,
            "annualization": args.annualization,
            "risk_free_rate": args.risk_free_rate,
            "bucket_columns": BUCKET_COLUMNS,
            "proxy_etfs": PROXY_ETFS,
            "benchmark_etf": BENCHMARK_ETF,
        },
        "results": averaged_rows,
        "period_results": comparison_rows,
        "model_details": detail_rows,
    }
    json_path.write_text(json.dumps(_json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8")

    print(frame.to_string(index=False))
    print(f"saved json: {json_path}")
    print(f"saved csv: {csv_path}")
    if args.random_periods:
        print(f"saved period csv: {period_csv_path}")
    print(f"saved md: {md_path}")


if __name__ == "__main__":
    main()
