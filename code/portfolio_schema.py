from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


CATEGORICAL_COLUMNS = [
    "OCCAT1",
    "OCCAT2",
    "INDCAT",
    "LF",
    "HOUSECL",
    "EDCL",
    "EDUC",
    "AGECL",
    "LIFECL",
    "FAMSTRUCT",
    "KIDS",
    "MARRIED",
    "EXPENSHILO",
    "WSAVED",
    "SAVRES1",
    "SAVRES2",
    "SAVRES3",
    "SAVRES4",
    "SAVRES5",
    "SAVRES6",
    "SAVRES7",
    "SAVRES8",
    "SAVRES9",
]

BUCKET_COLUMNS = [
    "cash_eq",
    "taxable_equity",
    "taxable_bond",
    "retirement_equity",
    "retirement_safe",
    "other_fin",
]

RISKY_BUCKET_COLUMNS = [
    "taxable_equity",
    "retirement_equity",
]

RISK_LABEL_BINS = [-np.inf, 0.05, 0.2, 0.4, 0.6, np.inf]


@dataclass
class PortfolioBuildResult:
    categorical_frame: pd.DataFrame
    allocation_frame: pd.DataFrame
    labels: pd.Series
    quality_frame: pd.DataFrame


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _safe_simplex(frame: pd.DataFrame) -> pd.DataFrame:
    row_sum = frame.sum(axis=1)
    safe_sum = row_sum.replace(0.0, np.nan)
    simplex = frame.div(safe_sum, axis=0).fillna(0.0)
    return simplex


def risky_share_from_allocation_frame(allocation_frame: pd.DataFrame) -> pd.Series:
    return allocation_frame[RISKY_BUCKET_COLUMNS].sum(axis=1)


def derive_risk_labels_from_allocation_frame(allocation_frame: pd.DataFrame) -> pd.Series:
    risky_share = risky_share_from_allocation_frame(allocation_frame)
    labels = pd.cut(
        risky_share,
        bins=RISK_LABEL_BINS,
        labels=[0, 1, 2, 3, 4],
        include_lowest=True,
    )
    return labels.astype(int)


def derive_risk_label_from_allocation_vector(allocation: np.ndarray) -> int:
    allocation_series = pd.Series(allocation, index=BUCKET_COLUMNS, dtype=float)
    risky_share = float(allocation_series[RISKY_BUCKET_COLUMNS].sum())
    labels = pd.cut(
        pd.Series([risky_share]),
        bins=RISK_LABEL_BINS,
        labels=[0, 1, 2, 3, 4],
        include_lowest=True,
    )
    return int(labels.iloc[0])


def _derive_labels(_: pd.DataFrame, allocation_frame: pd.DataFrame) -> pd.Series:
    return derive_risk_labels_from_allocation_frame(allocation_frame)


def build_non_overlapping_buckets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fin = _numeric(df, "FIN").clip(lower=0.0)
    cash_eq = (
        _numeric(df, "CHECKING")
        + _numeric(df, "SAVING")
        + _numeric(df, "MMA")
        + _numeric(df, "CDS")
    ).clip(lower=0.0)
    taxable_equity = _numeric(df, "DEQ").clip(lower=0.0)
    taxable_bond = _numeric(df, "BOND").clip(lower=0.0)
    retirement_equity = _numeric(df, "RETEQ").clip(lower=0.0)
    retirement_safe = (_numeric(df, "RETQLIQ") - retirement_equity).clip(lower=0.0)

    base = pd.DataFrame(
        {
            "cash_eq": cash_eq,
            "taxable_equity": taxable_equity,
            "taxable_bond": taxable_bond,
            "retirement_equity": retirement_equity,
            "retirement_safe": retirement_safe,
        }
    )

    accounted_before = base.sum(axis=1)
    overlap_scale = np.where(accounted_before > 0.0, np.minimum(1.0, fin / accounted_before), 1.0)
    scaled = base.mul(overlap_scale, axis=0)
    accounted_after = scaled.sum(axis=1)
    other_fin = (fin - accounted_after).clip(lower=0.0)
    buckets = scaled.assign(other_fin=other_fin)
    buckets = buckets[BUCKET_COLUMNS]

    quality = pd.DataFrame(
        {
            "FIN": fin,
            "accounted_before_rescale": accounted_before,
            "accounted_after_rescale": accounted_after,
            "coverage_before_rescale": np.where(fin > 0.0, accounted_before / fin, 0.0),
            "coverage_after_rescale": np.where(fin > 0.0, (accounted_after + other_fin) / fin, 0.0),
            "rescaled_overlap_flag": accounted_before > fin,
            "zero_fin_flag": fin <= 0.0,
            "other_fin_value": other_fin,
        }
    )
    return buckets, quality


def build_allocation_dataset(df: pd.DataFrame) -> PortfolioBuildResult:
    clean = df.copy()
    categorical = clean[CATEGORICAL_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    buckets, quality = build_non_overlapping_buckets(clean)
    simplex = _safe_simplex(buckets)
    labels = _derive_labels(clean, simplex)
    return PortfolioBuildResult(
        categorical_frame=categorical,
        allocation_frame=simplex,
        labels=labels,
        quality_frame=quality,
    )


def fit_allocation_clusters(
    train_allocation: pd.DataFrame,
    num_clusters: int = 12,
    random_state: int = 42,
) -> KMeans:
    model = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=20)
    model.fit(train_allocation.values)
    return model


def summarize_processed_split(
    allocation_frame: pd.DataFrame,
    labels: pd.Series,
    quality_frame: pd.DataFrame,
    cluster_ids: np.ndarray,
) -> Dict[str, object]:
    return {
        "num_rows": int(len(allocation_frame)),
        "allocation_columns": BUCKET_COLUMNS,
        "label_distribution": labels.value_counts().sort_index().to_dict(),
        "cluster_distribution": pd.Series(cluster_ids).value_counts().sort_index().to_dict(),
        "allocation_mean": allocation_frame.mean().round(6).to_dict(),
        "allocation_std": allocation_frame.std().round(6).to_dict(),
        "overlap_rescaled_share": float(quality_frame["rescaled_overlap_flag"].mean()),
        "zero_fin_share": float(quality_frame["zero_fin_flag"].mean()),
    }


def save_summary(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
