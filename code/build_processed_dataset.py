from __future__ import annotations

import argparse
from pathlib import Path

import torch
import pandas as pd

from portfolio_schema import (
    CATEGORICAL_COLUMNS,
    build_allocation_dataset,
    fit_allocation_clusters,
    save_summary,
    summarize_processed_split,
)


def build_split(
    split_name: str,
    df: pd.DataFrame,
    cardinalities: list[int],
    cluster_model,
    processed_dir: Path,
) -> None:
    result = build_allocation_dataset(df)
    cluster_ids = cluster_model.predict(result.allocation_frame.values)

    x_cat_tensor = torch.tensor(result.categorical_frame.values, dtype=torch.long)
    x_alloc_tensor = torch.tensor(result.allocation_frame.values, dtype=torch.float32)
    labels_tensor = torch.tensor(result.labels.values, dtype=torch.long)
    clusters_tensor = torch.tensor(cluster_ids, dtype=torch.long)
    cardinalities_tensor = torch.tensor(cardinalities, dtype=torch.long)

    torch.save(x_cat_tensor, processed_dir / f"{split_name}_x_cat_tensor.pt")
    torch.save(x_alloc_tensor, processed_dir / f"{split_name}_x_alloc_tensor.pt")
    torch.save(x_alloc_tensor, processed_dir / f"{split_name}_x_ratio_tensor.pt")
    torch.save(labels_tensor, processed_dir / f"{split_name}_labels_tensor.pt")
    torch.save(clusters_tensor, processed_dir / f"{split_name}_cluster_tensor.pt")
    torch.save(cardinalities_tensor, processed_dir / f"{split_name}_cardinalities.pt")

    quality_path = processed_dir / f"{split_name}_quality.csv"
    result.quality_frame.to_csv(quality_path, index=False)

    summary = summarize_processed_split(
        allocation_frame=result.allocation_frame,
        labels=result.labels,
        quality_frame=result.quality_frame,
        cluster_ids=cluster_ids,
    )
    save_summary(processed_dir / f"{split_name}_summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("../dataset"))
    parser.add_argument("--processed-dir", type=Path, default=Path("../dataset/processed"))
    parser.add_argument("--num-clusters", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    processed_dir = args.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(dataset_dir / "train.csv")
    test_df = pd.read_csv(dataset_dir / "test.csv")

    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_result = build_allocation_dataset(combined_df)
    cardinalities = [int(combined_result.categorical_frame[col].max()) + 1 for col in CATEGORICAL_COLUMNS]
    train_result = build_allocation_dataset(train_df)
    cluster_model = fit_allocation_clusters(
        train_allocation=train_result.allocation_frame,
        num_clusters=args.num_clusters,
        random_state=args.seed,
    )

    build_split("train", train_df, cardinalities, cluster_model, processed_dir)
    build_split("test", test_df, cardinalities, cluster_model, processed_dir)

    cluster_payload = {
        "num_clusters": args.num_clusters,
        "seed": args.seed,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "cluster_centers": cluster_model.cluster_centers_.round(6).tolist(),
    }
    save_summary(processed_dir / "cluster_meta.json", cluster_payload)


if __name__ == "__main__":
    main()
