from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from checkpoint_utils import load_dual_encoder_checkpoint
from portfolio_schema import BUCKET_COLUMNS, CATEGORICAL_COLUMNS, risky_share_to_bucket

REPO_ROOT = Path(__file__).resolve().parents[1]


def _normalize_alloc(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float32), 1e-8, None)
    return values / values.sum(axis=1, keepdims=True)


def _js_divergence(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    p = _normalize_alloc(y_true)
    q = _normalize_alloc(y_pred)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    midpoint = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / midpoint), axis=1)
    kl_qm = np.sum(q * np.log(q / midpoint), axis=1)
    return float(np.mean(0.5 * (kl_pm + kl_qm)))


def _risk_buckets(values: np.ndarray) -> np.ndarray:
    return np.array([risky_share_to_bucket(float(value)) for value in values], dtype=np.int64)


def evaluate_predictions(
    name: str,
    y_alloc_true: np.ndarray,
    y_alloc_pred: np.ndarray,
    y_risky_true: np.ndarray,
    y_risky_pred: np.ndarray,
) -> dict[str, object]:
    y_alloc_pred = _normalize_alloc(y_alloc_pred)
    y_risky_pred = np.clip(np.asarray(y_risky_pred, dtype=np.float32), 0.0, 1.0)
    true_bucket = _risk_buckets(y_risky_true)
    pred_bucket = _risk_buckets(y_risky_pred)

    risky_corr = np.corrcoef(y_risky_true, y_risky_pred)[0, 1]
    if not np.isfinite(risky_corr):
        risky_corr = 0.0

    return {
        "model": name,
        "allocation_mae": float(np.mean(np.abs(y_alloc_true - y_alloc_pred))),
        "allocation_rmse": float(np.sqrt(np.mean((y_alloc_true - y_alloc_pred) ** 2))),
        "allocation_js": _js_divergence(y_alloc_true, y_alloc_pred),
        "allocation_l1": float(np.mean(np.sum(np.abs(y_alloc_true - y_alloc_pred), axis=1))),
        "risky_share_mae": float(np.mean(np.abs(y_risky_true - y_risky_pred))),
        "risky_share_rmse": float(np.sqrt(np.mean((y_risky_true - y_risky_pred) ** 2))),
        "risky_share_corr": float(risky_corr),
        "risk_acc": float(accuracy_score(true_bucket, pred_bucket)),
        "risk_macro_f1": float(f1_score(true_bucket, pred_bucket, average="macro")),
        "risk_weighted_f1": float(f1_score(true_bucket, pred_bucket, average="weighted")),
        "confusion_matrix": confusion_matrix(true_bucket, pred_bucket).tolist(),
    }


class MeanAllocationBaseline:
    def fit(self, y_alloc: np.ndarray, y_risky: np.ndarray) -> "MeanAllocationBaseline":
        self.mean_alloc = _normalize_alloc(y_alloc.mean(axis=0, keepdims=True))[0]
        self.mean_risky = float(np.mean(y_risky))
        return self

    def predict(self, x_cat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n_rows = len(x_cat)
        return (
            np.tile(self.mean_alloc, (n_rows, 1)),
            np.full(n_rows, self.mean_risky, dtype=np.float32),
        )


class GroupMeanAllocationBaseline:
    def __init__(self, group_cols: list[str], min_count: int = 30):
        self.group_cols = group_cols
        self.min_count = min_count

    def fit(
        self,
        x_cat: pd.DataFrame,
        y_alloc: np.ndarray,
        y_risky: np.ndarray,
    ) -> "GroupMeanAllocationBaseline":
        self.global_alloc = _normalize_alloc(y_alloc.mean(axis=0, keepdims=True))[0]
        self.global_risky = float(np.mean(y_risky))
        alloc_cols = [f"alloc_{idx}" for idx in range(y_alloc.shape[1])]
        frame = x_cat[self.group_cols].copy()
        for idx, col in enumerate(alloc_cols):
            frame[col] = y_alloc[:, idx]
        frame["risky_share"] = y_risky

        grouped = frame.groupby(self.group_cols, dropna=False)
        table = grouped[alloc_cols + ["risky_share"]].mean()
        table["count"] = grouped.size()
        self.group_table = table[table["count"] >= self.min_count]
        return self

    def _predict_row(self, row: pd.Series) -> tuple[np.ndarray, float]:
        key = tuple(row[col] for col in self.group_cols)
        if len(self.group_cols) == 1:
            key = key[0]
        if key not in self.group_table.index:
            return self.global_alloc, self.global_risky

        record = self.group_table.loc[key]
        alloc = record[[f"alloc_{idx}" for idx in range(len(BUCKET_COLUMNS))]].to_numpy(dtype=np.float32)
        risky = float(record["risky_share"])
        return _normalize_alloc(alloc.reshape(1, -1))[0], risky

    def predict(self, x_cat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        allocs = []
        risky = []
        for _, row in x_cat.iterrows():
            alloc, risky_share = self._predict_row(row)
            allocs.append(alloc)
            risky.append(risky_share)
        return np.vstack(allocs), np.asarray(risky, dtype=np.float32)


class CatBoostAllocationBaseline:
    def __init__(self, cat_features: list[int], iterations: int, seed: int):
        self.cat_features = cat_features
        self.iterations = iterations
        self.seed = seed
        self.alloc_models: list[CatBoostRegressor] = []
        self.risky_model: CatBoostRegressor | None = None

    def _params(self) -> dict[str, object]:
        return {
            "iterations": self.iterations,
            "depth": 6,
            "learning_rate": 0.03,
            "loss_function": "MAE",
            "verbose": False,
            "random_seed": self.seed,
            "early_stopping_rounds": 50,
            "allow_writing_files": False,
        }

    def fit(
        self,
        x_train: pd.DataFrame,
        y_alloc_train: np.ndarray,
        y_risky_train: np.ndarray,
        x_valid: pd.DataFrame,
        y_alloc_valid: np.ndarray,
        y_risky_valid: np.ndarray,
    ) -> "CatBoostAllocationBaseline":
        self.alloc_models = []
        for idx in range(y_alloc_train.shape[1]):
            model = CatBoostRegressor(**self._params())
            model.fit(
                x_train,
                y_alloc_train[:, idx],
                cat_features=self.cat_features,
                eval_set=(x_valid, y_alloc_valid[:, idx]),
            )
            self.alloc_models.append(model)

        self.risky_model = CatBoostRegressor(**self._params())
        self.risky_model.fit(
            x_train,
            y_risky_train,
            cat_features=self.cat_features,
            eval_set=(x_valid, y_risky_valid),
        )
        return self

    def predict(self, x_cat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        alloc = np.column_stack([model.predict(x_cat) for model in self.alloc_models])
        risky = self.risky_model.predict(x_cat)
        return _normalize_alloc(alloc), np.clip(risky, 0.0, 1.0)


def source_encoder_predictions(
    checkpoint_dir: Path,
    prefix: str,
    processed_dir: Path,
    x_cat_tensor: torch.Tensor,
    y_alloc: np.ndarray,
    labels: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    cardinalities = torch.load(processed_dir / "test_cardinalities.pt").tolist()
    source_encoder, _, _, _ = load_dual_encoder_checkpoint(
        checkpoint_dir=checkpoint_dir,
        device=device,
        current_categorical_cols=CATEGORICAL_COLUMNS,
        current_categorical_cardinalities=cardinalities,
        current_x_cat_tensor=x_cat_tensor,
        current_x_ratio_tensor=torch.tensor(y_alloc, dtype=torch.float32),
        current_labels_tensor=labels,
        current_split="test",
        strict=False,
        prefix=prefix,
    )
    with torch.no_grad():
        output = source_encoder(x_cat_tensor.to(device))
    return output.allocation_probs.cpu().numpy(), output.risky_share.squeeze(1).cpu().numpy()


def source_encoder_knn_predictions(
    train_x_cat: np.ndarray,
    test_x_cat: np.ndarray,
    train_alloc: np.ndarray,
    train_risky: np.ndarray,
    source_alloc: np.ndarray,
    source_risky: np.ndarray,
    cardinalities: list[int],
    *,
    k: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    categories = [np.arange(cardinality) for cardinality in cardinalities]
    try:
        encoder = OneHotEncoder(categories=categories, sparse_output=False, handle_unknown="ignore")
    except TypeError:  # pragma: no cover
        encoder = OneHotEncoder(categories=categories, sparse=False, handle_unknown="ignore")
    train_onehot = encoder.fit_transform(train_x_cat)
    test_onehot = encoder.transform(test_x_cat)
    distances = cosine_distances(test_onehot, train_onehot)
    topk = np.argsort(distances, axis=1)[:, :k]
    anchor_alloc = train_alloc[topk].mean(axis=1)
    anchor_risky = train_risky[topk].mean(axis=1)
    alloc = alpha * source_alloc + (1.0 - alpha) * anchor_alloc
    risky = alpha * source_risky + (1.0 - alpha) * anchor_risky
    return _normalize_alloc(alloc), np.clip(risky, 0.0, 1.0)


def _load_split(processed_dir: Path, split: str) -> tuple[torch.Tensor, np.ndarray, np.ndarray, torch.Tensor]:
    x_cat = torch.load(processed_dir / f"{split}_x_cat_tensor.pt")
    x_alloc = torch.load(processed_dir / f"{split}_x_alloc_tensor.pt").numpy()
    risky = torch.load(processed_dir / f"{split}_risky_share_tensor.pt").squeeze(1).numpy()
    labels = torch.load(processed_dir / f"{split}_labels_tensor.pt")
    return x_cat, x_alloc, risky, labels


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "dataset" / "processed")
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--prefix", type=str, default="allocation_best")
    parser.add_argument(
        "--source-prefixes",
        type=str,
        default=None,
        help="Comma-separated label=prefix entries for multiple Source Encoder checkpoints.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "checkpoints" / "baseline_comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--catboost-iterations", type=int, default=300)
    parser.add_argument("--group-min-count", type=int, default=30)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--knn-alpha", type=float, default=0.7)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_x_cat_t, train_alloc, train_risky, _ = _load_split(args.processed_dir, "train")
    test_x_cat_t, test_alloc, test_risky, test_labels_t = _load_split(args.processed_dir, "test")
    cardinalities = torch.load(args.processed_dir / "train_cardinalities.pt").tolist()

    train_x_cat = pd.DataFrame(train_x_cat_t.numpy(), columns=CATEGORICAL_COLUMNS).astype(str)
    test_x_cat = pd.DataFrame(test_x_cat_t.numpy(), columns=CATEGORICAL_COLUMNS).astype(str)

    rows: list[dict[str, object]] = []

    mean_model = MeanAllocationBaseline().fit(train_alloc, train_risky)
    pred_alloc, pred_risky = mean_model.predict(test_x_cat)
    rows.append(evaluate_predictions("mean_allocation", test_alloc, pred_alloc, test_risky, pred_risky))

    group_specs = [
        ["AGECL"],
        ["AGECL", "HOUSECL"],
        ["AGECL", "HOUSECL", "EDCL"],
        ["AGECL", "HOUSECL", "EDCL", "OCCAT1"],
    ]
    for group_cols in group_specs:
        model = GroupMeanAllocationBaseline(group_cols, min_count=args.group_min_count).fit(
            train_x_cat,
            train_alloc,
            train_risky,
        )
        pred_alloc, pred_risky = model.predict(test_x_cat)
        rows.append(
            evaluate_predictions(
                "group_mean_" + "_".join(group_cols),
                test_alloc,
                pred_alloc,
                test_risky,
                pred_risky,
            )
        )

    train_idx, valid_idx = train_test_split(
        np.arange(len(train_x_cat)),
        test_size=0.15,
        random_state=args.seed,
        shuffle=True,
    )
    catboost_model = CatBoostAllocationBaseline(
        cat_features=list(range(len(CATEGORICAL_COLUMNS))),
        iterations=args.catboost_iterations,
        seed=args.seed,
    ).fit(
        train_x_cat.iloc[train_idx],
        train_alloc[train_idx],
        train_risky[train_idx],
        train_x_cat.iloc[valid_idx],
        train_alloc[valid_idx],
        train_risky[valid_idx],
    )
    pred_alloc, pred_risky = catboost_model.predict(test_x_cat)
    rows.append(evaluate_predictions("catboost", test_alloc, pred_alloc, test_risky, pred_risky))

    if args.source_prefixes:
        source_prefixes = []
        for item in args.source_prefixes.split(","):
            label, prefix = item.split("=", maxsplit=1)
            source_prefixes.append((label.strip(), prefix.strip()))
    else:
        source_prefixes = [("source_encoder", args.prefix)]

    for source_label, source_prefix in source_prefixes:
        source_alloc, source_risky = source_encoder_predictions(
            args.checkpoint_dir,
            source_prefix,
            args.processed_dir,
            test_x_cat_t,
            test_alloc,
            test_labels_t,
            device,
        )
        rows.append(evaluate_predictions(source_label, test_alloc, source_alloc, test_risky, source_risky))

        knn_alloc, knn_risky = source_encoder_knn_predictions(
            train_x_cat_t.numpy(),
            test_x_cat_t.numpy(),
            train_alloc,
            train_risky,
            source_alloc,
            source_risky,
            cardinalities,
            k=args.knn_k,
            alpha=args.knn_alpha,
        )
        rows.append(
            evaluate_predictions(
                f"{source_label}_knn_k{args.knn_k}_alpha{args.knn_alpha:g}",
                test_alloc,
                knn_alloc,
                test_risky,
                knn_risky,
            )
        )

    report = {
        "config": {
            "checkpoint_prefix": args.prefix,
            "source_prefixes": source_prefixes,
            "catboost_iterations": args.catboost_iterations,
            "group_min_count": args.group_min_count,
            "knn_k": args.knn_k,
            "knn_alpha": args.knn_alpha,
            "bucket_columns": BUCKET_COLUMNS,
        },
        "results": rows,
    }

    json_path = args.output_dir / "baseline_comparison.json"
    csv_path = args.output_dir / "baseline_comparison.csv"
    md_path = args.output_dir / "baseline_comparison.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    frame = pd.DataFrame([{k: v for k, v in row.items() if k != "confusion_matrix"} for row in rows])
    frame = frame.sort_values("allocation_mae")
    frame.to_csv(csv_path, index=False)
    md_path.write_text(_frame_to_markdown(frame), encoding="utf-8")

    print(frame.to_string(index=False))
    print(f"saved json: {json_path}")
    print(f"saved csv: {csv_path}")
    print(f"saved md: {md_path}")


if __name__ == "__main__":
    main()
