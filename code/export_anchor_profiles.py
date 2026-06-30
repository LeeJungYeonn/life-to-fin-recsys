from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

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


def row_to_profile(row: pd.Series, *, top_k: int, allow_cma: bool) -> dict:
    categorical_features = {
        column: int(pd.to_numeric(row[column], errors="coerce")) for column in CATEGORICAL_COLUMNS
    }
    return {
        "categorical_features": categorical_features,
        "options": {
            "top_k": top_k,
            "allow_cma": allow_cma,
        },
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=repo_root / "dataset" / "labeled" / "anchor_labeled_data.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "dataset" / "examples" / "anchor_profiles",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--allow-cma", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.limit is not None:
        df = df.head(args.limit)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        case_id = int(row["CASEID"])
        payload = row_to_profile(row, top_k=args.top_k, allow_cma=args.allow_cma)
        output_path = args.output_dir / f"case_{case_id}.json"
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"exported {len(df)} profile json file(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
