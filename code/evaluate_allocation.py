from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from checkpoint_utils import load_dual_encoder_checkpoint
from contrastive_utils import ordinal_logits_to_label
from portfolio_schema import CATEGORICAL_COLUMNS


def evaluate_linear_probe(features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    acc_scores = []
    f1_scores = []
    for train_idx, test_idx in cv.split(features, labels):
        model = clone(clf)
        model.fit(features[train_idx], labels[train_idx])
        pred = model.predict(features[test_idx])
        acc_scores.append(accuracy_score(labels[test_idx], pred))
        f1_scores.append(f1_score(labels[test_idx], pred, average="macro"))
    return {
        "accuracy_mean": float(np.mean(acc_scores)),
        "accuracy_std": float(np.std(acc_scores)),
        "macro_f1_mean": float(np.mean(f1_scores)),
        "macro_f1_std": float(np.std(f1_scores)),
    }


def domain_classifier_score(features: np.ndarray, modality_labels: np.ndarray) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    scores = []
    for train_idx, test_idx in cv.split(features, modality_labels):
        model = clone(clf)
        model.fit(features[train_idx], modality_labels[train_idx])
        pred = model.predict(features[test_idx])
        scores.append(accuracy_score(modality_labels[test_idx], pred))
    return {
        "accuracy_mean": float(np.mean(scores)),
        "accuracy_std": float(np.std(scores)),
    }


def retrieval_metrics(src: np.ndarray, tgt: np.ndarray, labels: np.ndarray, clusters: np.ndarray) -> dict[str, float]:
    sim = src @ tgt.T
    order = np.argsort(-sim, axis=1)
    ranked_labels = labels[order]
    ranked_clusters = clusters[order]

    label_matches = ranked_labels == labels[:, None]
    cluster_matches = ranked_clusters == clusters[:, None]
    exact_matches = order == np.arange(len(src))[:, None]

    def summarize(match_matrix: np.ndarray, prefix: str) -> dict[str, float]:
        first_hit = np.argmax(match_matrix, axis=1)
        has_hit = match_matrix.any(axis=1)
        payload = {
            f"{prefix}_recall@1": float(match_matrix[:, :1].any(axis=1).mean()),
            f"{prefix}_recall@5": float(match_matrix[:, :5].any(axis=1).mean()),
            f"{prefix}_recall@10": float(match_matrix[:, :10].any(axis=1).mean()),
            f"{prefix}_mrr": float(np.where(has_hit, 1.0 / (first_hit + 1), 0.0).mean()),
        }
        return payload

    metrics = {}
    metrics.update(summarize(exact_matches, "pair"))
    metrics.update(summarize(label_matches, "label"))
    metrics.update(summarize(cluster_matches, "cluster"))
    return metrics


def centroid_distance_by_class(src: np.ndarray, tgt: np.ndarray, labels: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for label in np.unique(labels):
        src_centroid = src[labels == label].mean(axis=0)
        tgt_centroid = tgt[labels == label].mean(axis=0)
        euclidean = float(np.linalg.norm(src_centroid - tgt_centroid))
        cosine = float(
            1
            - np.dot(src_centroid, tgt_centroid)
            / (np.linalg.norm(src_centroid) * np.linalg.norm(tgt_centroid) + 1e-8)
        )
        rows.append(
            {
                "risk_label": int(label),
                "euclidean_distance": euclidean,
                "cosine_distance": cosine,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("../dataset/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("../checkpoints"))
    parser.add_argument("--prefix", type=str, default="allocation_best")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--report-path", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_cat = torch.load(args.processed_dir / f"{args.split}_x_cat_tensor.pt")
    x_alloc = torch.load(args.processed_dir / f"{args.split}_x_alloc_tensor.pt")
    labels = torch.load(args.processed_dir / f"{args.split}_labels_tensor.pt")
    clusters = torch.load(args.processed_dir / f"{args.split}_cluster_tensor.pt")
    cardinalities = torch.load(args.processed_dir / f"{args.split}_cardinalities.pt").tolist()

    source_encoder, target_encoder, checkpoint_meta, validation = load_dual_encoder_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        current_categorical_cols=CATEGORICAL_COLUMNS,
        current_categorical_cardinalities=cardinalities,
        current_x_cat_tensor=x_cat,
        current_x_ratio_tensor=x_alloc,
        current_labels_tensor=labels,
        current_split=args.split,
        strict=False,
        prefix=args.prefix,
    )

    with torch.no_grad():
        source_output = source_encoder(x_cat.to(device))
        target_output = target_encoder(x_alloc.to(device))

    source_probs = source_output.allocation_probs.cpu()
    target_probs = target_output.allocation_probs.cpu()
    source_pred = ordinal_logits_to_label(source_output.risk_logits).cpu().numpy()
    target_pred = ordinal_logits_to_label(target_output.risk_logits).cpu().numpy()
    labels_np = labels.numpy()
    clusters_np = clusters.numpy()

    z_source = F.normalize(source_output.embedding.cpu(), dim=1).numpy()
    z_target = F.normalize(target_output.embedding.cpu(), dim=1).numpy()

    concat_z = np.concatenate([z_source, z_target], axis=0)
    concat_labels = np.concatenate([labels_np, labels_np], axis=0)
    concat_modality = np.array([0] * len(z_source) + [1] * len(z_target))

    report = {
        "split": args.split,
        "num_examples": int(len(labels_np)),
        "checkpoint_prefix": args.prefix,
        "validation_warnings": validation["warnings"],
        "source_risk_acc": float(accuracy_score(labels_np, source_pred)),
        "source_risk_macro_f1": float(f1_score(labels_np, source_pred, average="macro")),
        "target_risk_acc": float(accuracy_score(labels_np, target_pred)),
        "target_risk_macro_f1": float(f1_score(labels_np, target_pred, average="macro")),
        "source_alloc_mae": float(torch.mean(torch.abs(source_probs - x_alloc)).item()),
        "target_alloc_mae": float(torch.mean(torch.abs(target_probs - x_alloc)).item()),
        "source_confusion_matrix": confusion_matrix(labels_np, source_pred).tolist(),
        "retrieval": retrieval_metrics(z_source, z_target, labels_np, clusters_np),
        "probe": {
            "source": evaluate_linear_probe(z_source, labels_np),
            "target": evaluate_linear_probe(z_target, labels_np),
        },
        "domain_classifier": domain_classifier_score(concat_z, concat_modality),
        "silhouette": {
            "risk_cosine": float(silhouette_score(concat_z, concat_labels, metric="cosine")),
            "modality_cosine": float(silhouette_score(concat_z, concat_modality, metric="cosine")),
        },
        "centroid_distance_by_class": centroid_distance_by_class(z_source, z_target, labels_np),
    }

    report_path = args.report_path or args.checkpoint_dir / f"{args.prefix}_{args.split}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
