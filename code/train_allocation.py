from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from checkpoint_utils import (
    build_preprocess_info,
    capture_rng_state,
    make_torch_generator,
    save_dual_encoder_checkpoint,
    set_reproducible_mode,
)
from contrastive_utils import (
    build_cross_modal_positive_mask,
    continuous_portfolio_loss,
    coral_loss,
    multi_positive_supcon_loss,
    ordinal_logits_to_label,
    ordinal_regression_loss,
)
from models import SourceEncoder, TargetEncoder
from portfolio_schema import CATEGORICAL_COLUMNS


class AllocationDataset(Dataset):
    def __init__(self, x_cat, x_alloc, labels, clusters):
        self.x_cat = x_cat
        self.x_alloc = x_alloc
        self.labels = labels
        self.clusters = clusters

    def __len__(self):
        return len(self.x_cat)

    def __getitem__(self, idx):
        return self.x_cat[idx], self.x_alloc[idx], self.labels[idx], self.clusters[idx]


def evaluate_model(source_encoder, target_encoder, dataloader, device):
    source_encoder.eval()
    target_encoder.eval()

    total_abs_error = 0.0
    total_js = 0.0
    total_examples = 0
    source_preds = []
    target_preds = []
    labels_all = []

    with torch.no_grad():
        for batch_x_cat, batch_x_alloc, batch_labels, _ in dataloader:
            batch_x_cat = batch_x_cat.to(device)
            batch_x_alloc = batch_x_alloc.to(device)
            batch_labels = batch_labels.to(device)

            source_output = source_encoder(batch_x_cat)
            target_output = target_encoder(batch_x_alloc)

            batch_size = batch_x_cat.size(0)
            source_probs = source_output.allocation_probs
            total_abs_error += torch.abs(source_probs - batch_x_alloc).sum().item()
            total_examples += batch_size * batch_x_alloc.size(1)
            total_js += batch_size * continuous_portfolio_loss(
                source_output.allocation_logits,
                batch_x_alloc,
            )[1]["js"]

            source_preds.append(ordinal_logits_to_label(source_output.risk_logits).cpu())
            target_preds.append(ordinal_logits_to_label(target_output.risk_logits).cpu())
            labels_all.append(batch_labels.cpu())

    source_preds = torch.cat(source_preds).numpy()
    target_preds = torch.cat(target_preds).numpy()
    labels_np = torch.cat(labels_all).numpy()

    return {
        "source_alloc_mae": float(total_abs_error / max(total_examples, 1)),
        "source_alloc_js": float(total_js / max(len(labels_np), 1)),
        "source_risk_acc": float(accuracy_score(labels_np, source_preds)),
        "source_risk_macro_f1": float(f1_score(labels_np, source_preds, average="macro")),
        "target_risk_acc": float(accuracy_score(labels_np, target_preds)),
        "target_risk_macro_f1": float(f1_score(labels_np, target_preds, average="macro")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("../dataset/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("../checkpoints"))
    parser.add_argument("--prefix", type=str, default="allocation_best")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--target-input-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--js-threshold", type=float, default=0.08)
    args = parser.parse_args()

    set_reproducible_mode(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_cat = torch.load(args.processed_dir / "train_x_cat_tensor.pt")
    x_alloc = torch.load(args.processed_dir / "train_x_alloc_tensor.pt")
    labels = torch.load(args.processed_dir / "train_labels_tensor.pt")
    clusters = torch.load(args.processed_dir / "train_cluster_tensor.pt")
    cardinalities = torch.load(args.processed_dir / "train_cardinalities.pt").tolist()

    input_dim = args.target_input_dim or x_alloc.shape[1]
    dataset = AllocationDataset(x_cat, x_alloc, labels, clusters)

    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.validation_ratio,
        random_state=args.seed,
        shuffle=True,
        stratify=labels.numpy(),
    )
    train_dataset = Subset(dataset, train_idx.tolist())
    val_dataset = Subset(dataset, val_idx.tolist())

    generator = make_torch_generator(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    source_encoder = SourceEncoder(
        cardinalities,
        embed_dim=16,
        output_dim=args.output_dim,
        projection_dim=args.projection_dim,
        num_risk_levels=5,
        allocation_dim=input_dim,
    ).to(device)
    target_encoder = TargetEncoder(
        input_dim=input_dim,
        output_dim=args.output_dim,
        projection_dim=args.projection_dim,
        num_risk_levels=5,
        allocation_dim=input_dim,
    ).to(device)

    optimizer = optim.AdamW(
        list(source_encoder.parameters()) + list(target_encoder.parameters()),
        lr=args.learning_rate,
    )

    loss_weights = {
        "supcon": 0.4,
        "source_alloc": 2.0,
        "target_alloc": 0.5,
        "source_risk": 0.2,
        "target_risk": 0.1,
        "coral": 0.05,
    }

    best_state = None
    best_metrics = None
    best_mae = float("inf")
    epochs_without_improve = 0

    preprocess_info = build_preprocess_info(
        categorical_cols=CATEGORICAL_COLUMNS,
        categorical_cardinalities=cardinalities,
        x_cat_tensor=x_cat,
        x_ratio_tensor=x_alloc,
        labels_tensor=labels,
        split="train",
    )

    for epoch in range(args.epochs):
        source_encoder.train()
        target_encoder.train()
        running = {
            "total": 0.0,
            "supcon": 0.0,
            "source_alloc": 0.0,
            "target_alloc": 0.0,
            "source_risk": 0.0,
            "target_risk": 0.0,
            "coral": 0.0,
        }

        for batch_x_cat, batch_x_alloc, batch_labels, batch_clusters in train_loader:
            batch_x_cat = batch_x_cat.to(device)
            batch_x_alloc = batch_x_alloc.to(device)
            batch_labels = batch_labels.to(device)
            batch_clusters = batch_clusters.to(device)

            optimizer.zero_grad(set_to_none=True)

            source_output = source_encoder(batch_x_cat)
            target_output = target_encoder(batch_x_alloc)

            positive_mask = build_cross_modal_positive_mask(
                batch_x_alloc,
                batch_x_alloc,
                labels=batch_labels,
                cluster_ids=batch_clusters,
                js_threshold=args.js_threshold,
            )
            embeddings = torch.cat([source_output.embedding, target_output.embedding], dim=0)
            loss_supcon = multi_positive_supcon_loss(
                embeddings,
                positive_mask=positive_mask,
                temperature=args.temperature,
            )

            source_alloc_loss, source_alloc_metrics = continuous_portfolio_loss(
                source_output.allocation_logits,
                batch_x_alloc,
            )
            target_alloc_loss, target_alloc_metrics = continuous_portfolio_loss(
                target_output.allocation_logits,
                batch_x_alloc,
            )
            loss_source_risk = ordinal_regression_loss(
                source_output.risk_logits,
                batch_labels,
                num_risk_levels=5,
            )
            loss_target_risk = ordinal_regression_loss(
                target_output.risk_logits,
                batch_labels,
                num_risk_levels=5,
            )
            loss_coral = coral_loss(source_output.hidden, target_output.hidden)

            total_loss = (
                loss_weights["supcon"] * loss_supcon
                + loss_weights["source_alloc"] * source_alloc_loss
                + loss_weights["target_alloc"] * target_alloc_loss
                + loss_weights["source_risk"] * loss_source_risk
                + loss_weights["target_risk"] * loss_target_risk
                + loss_weights["coral"] * loss_coral
            )
            total_loss.backward()
            optimizer.step()

            running["total"] += float(total_loss.detach().item())
            running["supcon"] += float(loss_supcon.detach().item())
            running["source_alloc"] += source_alloc_metrics["js"] + source_alloc_metrics["l1"]
            running["target_alloc"] += target_alloc_metrics["js"] + target_alloc_metrics["l1"]
            running["source_risk"] += float(loss_source_risk.detach().item())
            running["target_risk"] += float(loss_target_risk.detach().item())
            running["coral"] += float(loss_coral.detach().item())

        train_metrics = {key: value / len(train_loader) for key, value in running.items()}
        val_metrics = evaluate_model(source_encoder, target_encoder, val_loader, device)

        if val_metrics["source_alloc_mae"] < best_mae:
            best_mae = val_metrics["source_alloc_mae"]
            best_metrics = {"epoch": epoch + 1, **train_metrics, **val_metrics}
            best_state = {
                "source": copy.deepcopy(source_encoder.state_dict()),
                "target": copy.deepcopy(target_encoder.state_dict()),
            }
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        print(
            f"epoch={epoch + 1} "
            f"train_total={train_metrics['total']:.4f} "
            f"val_alloc_mae={val_metrics['source_alloc_mae']:.4f} "
            f"val_alloc_js={val_metrics['source_alloc_js']:.4f} "
            f"val_risk_f1={val_metrics['source_risk_macro_f1']:.4f}"
        )

        if epochs_without_improve >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint state.")

    source_encoder.load_state_dict(best_state["source"])
    target_encoder.load_state_dict(best_state["target"])

    best_params = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "output_dim": args.output_dim,
        "projection_dim": args.projection_dim,
        "validation_ratio": args.validation_ratio,
        "loss_weights": loss_weights,
        "js_threshold": args.js_threshold,
    }
    save_dual_encoder_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        source_encoder=source_encoder,
        target_encoder=target_encoder,
        preprocess_info=preprocess_info,
        model_config={
            "embed_dim": 16,
            "output_dim": args.output_dim,
            "projection_dim": args.projection_dim,
            "num_risk_levels": 5,
            "ratio_dim": input_dim,
            "allocation_dim": input_dim,
            "target_input_dim": input_dim,
        },
        best_params=best_params,
        best_loss=best_mae,
        optimizer=optimizer,
        epoch=best_metrics["epoch"],
        rng_state=capture_rng_state(dataloader_generator=generator),
        seed=args.seed,
        deterministic=True,
        prefix=args.prefix,
    )

    metrics_path = args.checkpoint_dir / f"{args.prefix}_metrics.json"
    metrics_path.write_text(json.dumps(best_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved checkpoint prefix={args.prefix} at {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
