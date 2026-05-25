from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recsys.naverpay_catalog import load_default_snapshot, load_snapshot as load_naver_snapshot
from recsys.pykrx_catalog import load_snapshot as load_pykrx_snapshot
from recsys.ranker import UserRequest, recommend_products


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk-level", type=int, required=True, help="0=very conservative, 4=very aggressive")
    parser.add_argument(
        "--allocation",
        type=float,
        nargs=6,
        required=True,
        metavar=("CASH_EQ", "TAXABLE_EQUITY", "TAXABLE_BOND", "RET_EQUITY", "RET_SAFE", "OTHER_FIN"),
    )
    parser.add_argument("--allow-cma", action="store_true")
    parser.add_argument("--naverpay-path", type=Path, default=None)
    parser.add_argument("--pykrx-path", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    allocation = np.array(args.allocation, dtype=np.float32)
    allocation_sum = allocation.sum()
    if allocation_sum <= 0.0:
        raise ValueError("Allocation sum must be positive.")
    allocation = allocation / allocation_sum

    products = []
    if args.naverpay_path is not None:
        products.extend(load_naver_snapshot(args.naverpay_path))
    else:
        products.extend(load_default_snapshot())

    if args.pykrx_path is not None and args.pykrx_path.exists():
        products.extend(load_pykrx_snapshot(args.pykrx_path))

    result = recommend_products(
        products,
        UserRequest(
            risk_level=args.risk_level,
            predicted_allocation=allocation,
            allow_cma=args.allow_cma,
        ),
        top_k=args.top_k,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
