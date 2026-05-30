from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_list(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _run(command: list[str], cwd: Path) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _combo_prefix(base_prefix: str, combo: dict[str, object]) -> str:
    lr_token = str(combo["learning_rate"]).replace(".", "p").replace("-", "m")
    dropout_token = str(combo["dropout"]).replace(".", "p")
    return (
        f"{base_prefix}"
        f"_emb{combo['embed_dim']}"
        f"_out{combo['output_dim']}"
        f"_drop{dropout_token}"
        f"_lr{lr_token}"
        f"_bs{combo['batch_size']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "checkpoints" / "grid_search")
    parser.add_argument("--output-csv", type=Path, default=REPO_ROOT / "checkpoints" / "grid_search" / "grid_results.csv")
    parser.add_argument("--base-prefix", type=str, default="grid")
    parser.add_argument("--embed-dims", type=str, default="8,16,32")
    parser.add_argument("--output-dims", type=str, default="128,256,512")
    parser.add_argument("--dropouts", type=str, default="0.1,0.2,0.3")
    parser.add_argument("--learning-rates", type=str, default="0.0005,0.001,0.002")
    parser.add_argument("--batch-sizes", type=str, default="256,512,1024")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-test-eval", action="store_true")
    args = parser.parse_args()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    grid = {
        "embed_dim": _parse_list(args.embed_dims, int),
        "output_dim": _parse_list(args.output_dims, int),
        "dropout": _parse_list(args.dropouts, float),
        "learning_rate": _parse_list(args.learning_rates, float),
        "batch_size": _parse_list(args.batch_sizes, int),
    }
    combos = [
        dict(zip(grid.keys(), values))
        for values in itertools.product(*grid.values())
    ]
    if args.max_runs is not None:
        combos = combos[: args.max_runs]

    fieldnames = [
        "prefix",
        "embed_dim",
        "output_dim",
        "dropout",
        "learning_rate",
        "batch_size",
        "best_epoch",
        "val_alloc_mae",
        "val_alloc_js",
        "val_risky_share_mae",
        "val_risk_bucket_macro_f1",
        "test_alloc_mae",
        "test_risky_share_mae",
        "test_risk_bucket_macro_f1",
    ]
    rows = []

    for idx, combo in enumerate(combos, start=1):
        prefix = _combo_prefix(args.base_prefix, combo)
        metrics_path = args.checkpoint_dir / f"{prefix}_metrics.json"
        test_report_path = args.checkpoint_dir / f"{prefix}_test_report.json"
        print(f"[{idx}/{len(combos)}] {prefix}", flush=True)

        if not (args.skip_existing and metrics_path.exists()):
            train_cmd = [
                sys.executable,
                "code/train_allocation.py",
                "--checkpoint-dir",
                str(args.checkpoint_dir),
                "--prefix",
                prefix,
                "--seed",
                str(args.seed),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--embed-dim",
                str(combo["embed_dim"]),
                "--output-dim",
                str(combo["output_dim"]),
                "--dropout",
                str(combo["dropout"]),
                "--learning-rate",
                str(combo["learning_rate"]),
                "--batch-size",
                str(combo["batch_size"]),
            ]
            _run(train_cmd, REPO_ROOT)

        if not args.no_test_eval and not (args.skip_existing and test_report_path.exists()):
            eval_cmd = [
                sys.executable,
                "code/evaluate_allocation.py",
                "--checkpoint-dir",
                str(args.checkpoint_dir),
                "--prefix",
                prefix,
                "--split",
                "test",
                "--report-path",
                str(test_report_path),
            ]
            _run(eval_cmd, REPO_ROOT)

        metrics = _load_json(metrics_path)
        test_report = _load_json(test_report_path) if test_report_path.exists() else {}
        rows.append(
            {
                "prefix": prefix,
                **combo,
                "best_epoch": metrics.get("epoch"),
                "val_alloc_mae": metrics.get("source_alloc_mae"),
                "val_alloc_js": metrics.get("source_alloc_js"),
                "val_risky_share_mae": metrics.get("source_risky_share_mae"),
                "val_risk_bucket_macro_f1": metrics.get("source_risk_bucket_macro_f1"),
                "test_alloc_mae": test_report.get("source_alloc_mae"),
                "test_risky_share_mae": test_report.get("source_risky_share_mae"),
                "test_risk_bucket_macro_f1": test_report.get("source_risk_bucket_macro_f1"),
            }
        )

        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: row["val_alloc_mae"]))

    print(f"saved grid results: {args.output_csv}")


if __name__ == "__main__":
    main()
