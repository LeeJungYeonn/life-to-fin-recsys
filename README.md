# Life-to-Fin: Portfolio Recommendation via Demographic Data

**A cold-start, portfolio-aware financial product recommender that predicts asset allocation from demographic and lifestyle features—without requiring a user's transaction or investment history.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Model-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

**Project:** Recommendation System, Ewha Womans University  
**Team:** Jungyeon Lee · Seonyeong Jeong  
**[Final Presentation (PDF)](https://drive.google.com/file/d/1wzlaiiQp36cVmvfCpn4BgE2KRTilr4E1/view)**

## Overview

Conventional financial recommenders often depend on transaction histories, investment records, credit data, or self-reported risk questionnaires. This creates a **cold-start problem** for new investors and *thin-file* users who have little or no financial history.

**Life-to-Fin** explores a different starting point: learning patterns between a user's demographic/lifestyle characteristics and their observed financial asset allocation. The system predicts a four-asset portfolio, derives a risk profile, and recommends **specific financial products with suggested investment weights**.

The final output is not merely a risk label or an asset-class prediction. It is a **portfolio-aware basket of five products**, accompanied by an interpretable summary in a Streamlit demo.

### Key features

- **No transaction history required at inference:** uses 23 categorical inputs covering demographics, employment, housing, spending/saving summaries, and savings goals.
- **Cross-domain representation learning:** aligns non-financial source representations with financial target representations during training using contrastive learning.
- **Joint allocation and risk prediction:** estimates weights for cash, bonds, pension, and equity, together with a continuous risky-share score.
- **Hybrid product recommendation:** combines constraint-based filtering, content-based scoring, product–product graph score diffusion, and allocation-aware basket construction.
- **Concrete outputs:** recommends up to five products with individual portfolio weights and a descriptive report.

> **Scope:** This is a research/course project, not a production financial-advice service. The report describes the model's output; it should not be interpreted as a causal explanation or a guarantee of investment suitability.

---

## System Architecture

```text
                    Demographic & Lifestyle Profile
                       23 categorical features
                                 |
                          Embedding Lookup
                                 |
                      Source Encoder (MLP)
                                 |
                    Latent Financial Profile
                         /               \
                  Allocation Head     Risky-Share Head
                   (Softmax)             (Sigmoid)
                        |                    |
           [Cash, Bond, Pension, Equity]  Risk Bucket
                        \                    /
                         \                  /
                       Predicted User Portfolio
                                 |
                    Financial Product Catalog
                     (Naver Pay + PyKrx)
                                 |
                   Constraint-Based Filtering
                                 |
                     Content-Based Scoring
                                 |
                  Product–Product Graph Diffusion
                                 |
              Allocation-Aware Top-K Product Selection
                         + Weight Assignment
                                 |
                      Recommended Basket
                    + Streamlit Summary Report
```

**During training only**, a **Target Encoder** embeds the observed financial allocation. Contrastive alignment and allocation/risk prediction losses help train the Source Encoder. The Target Encoder is **not needed at inference time**.

### 1. Source Encoder & Allocation Prediction

SCF input variables are organized into **15 semantic feature groups**, represented by **23 categorical columns** in the implementation (including `SAVRES1`–`SAVRES9`).

| Component | Implementation |
|---|---|
| Input | 23 categorical features |
| Embeddings | 32 dimensions per feature (736 concatenated dimensions) |
| Source Encoder | MLP: `736 → 256 → 512` with BatchNorm, ReLU, and dropout |
| Projection head | Maps the source representation to a contrastive-learning space |
| Allocation head | `512 → 256 → 4` + Softmax |
| Risky-share head | `512 → 256 → 1` + Sigmoid |
| Allocation output | `[cash, bond, pension, equity]`, summing to 1 |

The training objective includes **allocation regression/distribution matching** (L1 and Jensen–Shannon divergence), **risky-share regression** (Smooth L1), and **contrastive alignment**. The financial Target Encoder acts as a training-time representation anchor.

The inference pipeline also supports **non-financial feature-space kNN smoothing**, which blends the model output with allocations and risky-share targets from similar anchor profiles. This can be disabled with `--disable-knn-smoothing`.

Relevant files: [`models.py`](code/models.py), [`train_allocation.py`](code/train_allocation.py), [`contrastive_utils.py`](code/contrastive_utils.py), [`profile_knn.py`](code/profile_knn.py).

### 2. Risk Level Derivation

A sigmoid head predicts a continuous **risky-share** estimate. The system maps it to a five-level risk bucket.

| Risk level | Profile | Predicted risky share |
|---|---|---|
| 1 | Very conservative | ≤ 0.15 |
| 2 | Conservative | > 0.15–0.35 |
| 3 | Moderate | > 0.35–0.55 |
| 4 | Aggressive | > 0.55–0.75 |
| 5 | Very aggressive | > 0.75 |

The CLI defaults to model-derived risk (`--risk-source model`); it can alternatively derive the bucket from allocation (`--risk-source allocation`). Internally, the recommender uses **zero-based** risk buckets (`0`–`4`); the demo displays the corresponding human-readable levels (`1`–`5`).

### 3. Portfolio-Aware Product Recommendation

The recommendation stage is a **constraint-based + content-based hybrid**, with graph-enhanced scoring and allocation-constrained basket construction.

1. **Candidate filtering:** apply relevant risk-context and user-option rules. For example, CMA eligibility depends on the `allow_cma` option, and certain equity-exposed ETFs are excluded when the target equity allocation is zero.
2. **Product scoring:** compute a base score from the product's **asset exposure/allocation match**, optional preference metadata, interest/yield information, principal-protection and deposit-insurance attributes, and liquidity features.
3. **Graph-based score diffusion:** refine scores over a **product–product similarity graph**. This is *item-content similarity*, not user–item interaction-based collaborative filtering.
4. **Allocation-aware basket construction:** select products from eligible asset buckets using graph-refined scores and assign within-bucket weights, aiming to reflect the predicted allocation.
5. **Output:** return the **ranked products** and the **optimized basket**, including product-level weights.

> The filter is **not** a universal rule requiring every recommended product's risk label to be less than or equal to the user's risk label. Product-level filtering and portfolio-level risk/allocation matching are distinct steps.

Relevant files: [`ranker.py`](code/recsys/ranker.py), [`graph_builder.py`](code/recsys/graph_builder.py), [`optimizer.py`](code/recsys/optimizer.py), [`run_end_to_end.py`](code/run_end_to_end.py).

---

## Data

### Household profiles: Survey of Consumer Finances (SCF)

The [Survey of Consumer Finances](https://www.federalreserve.gov/econres/scfindex.htm) provides U.S. household demographic and balance-sheet information. We use demographic/lifestyle variables as **source inputs** and observed financial holdings to construct **training targets**.

| Data role | Examples |
|---|---|
| Occupation and employment | `OCCAT1`, `OCCAT2`, `INDCAT`, `LF` |
| Housing and demographics | `HOUSECL`, `AGECL`, `LIFECL`, `EDCL`, `EDUC`, `FAMSTRUCT`, `KIDS`, `MARRIED` |
| Spending, saving, and goals | `EXPENSHILO`, `WSAVED`, `SAVRES1`–`SAVRES9` |
| Financial targets (training/evaluation only) | Cash-like assets, bonds, pension/retirement holdings, equity |

The 23 input variables include **summary spending/saving responses**, but **not transaction histories, account balances, or investment holdings**. Actual financial holdings are used to build labels during supervised training and evaluation, not as inputs for a new user.

### Product catalogs

- **Naver Pay Savings:** curated Korean deposit, installment-savings, parking-account, and CMA products, with rate, deposit-insurance, principal-protection, term, and liquidity attributes.
- **PyKrx:** Korean-market ETF snapshots with asset-class and subtype information, historical market data, volatility estimates, and product risk labels.

Product catalogs are loaded from local snapshots; recommendations are not presented as live quotations or continuously updated product availability.

---

## Evaluation

### Allocation prediction

The final presentation reports the following comparison against common tabular and allocation baselines (lower is better except risk accuracy):

| Model | MAE ↓ | RMSE ↓ | JS Divergence ↓ | Risky-share MAE ↓ | Risk Accuracy ↑ |
|---|---:|---:|---:|---:|---:|
| **Life-to-Fin** | **0.1719** | 0.2799 | **0.1304** | 0.2324 | 0.3104 |
| CatBoost | 0.1727 | 0.2839 | 0.1331 | **0.2253** | **0.3703** |
| Best demographic-group mean | 0.1947 | **0.2707** | 0.1405 | 0.2452 | 0.2212 |
| Global mean allocation | 0.2155 | 0.2873 | 0.1602 | 0.2622 | 0.1804 |

Our approach achieved the **lowest reported MAE and JS divergence**, while CatBoost performed better on risky-share MAE and risk classification accuracy, and the best group-mean baseline had the lowest RMSE. The results support the feasibility of demographic-based allocation prediction, **not uniformly superior performance across every metric**.

### Diversification

The recommended product basket is evaluated using **normalized Herfindahl–Hirschman Index (HHI)** diversification:

\[
D = \frac{1 - \sum_g p_g^2}{1 - 1/K}
\]

where \(p_g\) is the aggregate portfolio proportion of group \(g\), and \(K\) is the number of feasible groups for normalization. Scores closer to 1 indicate a more even distribution across the considered groups.

| Reported basket metric | Score |
|---|---:|
| Asset diversification | 0.521 |
| Category diversification | 0.766 |
| Subtype diversification | 0.767 |
| **Overall diversification** | **0.669** |

The overall score is defined as `0.4 × Asset Diversification + 0.6 × Subtype Diversification`. Category diversification is reported separately. These figures refer to the **basket reported in the presentation**, not a universal guarantee for every recommendation.

### Historical proxy-ETF backtest

To assess risk-adjusted performance, the project maps predicted asset-class weights to representative ETFs:

| Asset class | Proxy ETF |
|---|---|
| Cash | KODEX KOFR금리액티브(합성) |
| Bond | KODEX 국고채3년 |
| Equity | KODEX 200 |
| Pension | KODEX TDF2030액티브 적격 |

The presentation reports Sharpe ratios of **1.68–1.96** across the five predicted risk groups, compared with **1.02** for the 100% KODEX 200 benchmark in the selected historical periods. The recommended allocations had lower annualized returns but substantially lower volatility.

**Important:** this is a *historical, asset-class proxy backtest* using risk-bucket-average predicted allocations—not an execution-level backtest of the five recommended products, and not evidence of future returns. Results depend on the selected time windows and benchmark.

Relevant evaluation scripts: [`evaluate_baselines.py`](code/evaluate_baselines.py), [`diversification_metrics.py`](code/diversification_metrics.py), [`evaluate_proxy_etf_sharpe.py`](code/evaluate_proxy_etf_sharpe.py).

---

## Demo

The [Streamlit demo](demo/app.py) allows a user to enter a demographic/lifestyle profile or load a test `CASEID`. It displays:

- predicted asset allocation across four asset classes;
- a five-level risk profile;
- recommended products, scores, and portfolio weights;
- a profile-and-allocation summary report.

The report is based on **predefined descriptive rules applied to the input profile and predicted allocation**. It does not use an LLM and does not establish which individual input feature *caused* a prediction.

### Illustrative recommendation

In the final presentation, a test household (`CASEID 163706`) received **risk level 3** and the following recommended basket:

| Product | Asset bucket | Suggested weight |
|---|---|---:|
| JB 슈퍼씨드 적금 | Cash | 24.35% |
| TIGER TDF2045 적격 | Pension | 25.00% |
| KODEX TDF2050액티브 적격 | Pension | 24.99% |
| RISE TDF2050액티브 적격 | Pension | 24.98% |
| KODEX 200 | Equity | 0.68% |

*This is one demonstration case, not a general portfolio recommendation.*

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/LeeJungYeonn/life-to-fin-recsys.git
cd life-to-fin-recsys

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
pip install -r demo/requirements_for_demo.txt
```

### 2. Run the Streamlit demo

```bash
streamlit run demo/app.py
```

The demo loads the project's processed-data artifacts, available catalog snapshots, the trained `allocation_best_*` checkpoint, and anchor profiles. Ensure these files are present in their expected repository directories before running.

### 3. Run command-line inference

Create a `profile.json` containing **all 23 categorical features** (optionally nested under `"categorical_features"`). Input values must follow the SCF-derived encodings used during training. The input may also include an `"options"` object with `top_k` (default 5) and `allow_cma` (default `false`).

For example, the following SCF-coded profile uses a working, home-owning household aged 35–44, with a liquidity-oriented savings goal:

```json
{
  "categorical_features": {
    "OCCAT1": 1, "OCCAT2": 1, "INDCAT": 2, "LF": 1,
    "HOUSECL": 1, "EDCL": 4, "EDUC": 12, "AGECL": 2,
    "LIFECL": 1, "FAMSTRUCT": 2, "KIDS": 0, "MARRIED": 2,
    "EXPENSHILO": 3, "WSAVED": 3,
    "SAVRES1": 0, "SAVRES2": 0, "SAVRES3": 0,
    "SAVRES4": 0, "SAVRES5": 0, "SAVRES6": 0,
    "SAVRES7": 1, "SAVRES8": 0, "SAVRES9": 0
  },
  "options": {"top_k": 5, "allow_cma": false}
}
```

Save the example as `profile.json`, then run:

```bash
python code/run_end_to_end.py --input profile.json
```

Optional arguments include `--checkpoint-prefix`, `--risk-source {model,allocation}`, `--knn-k`, `--knn-alpha`, and `--disable-knn-smoothing`. The script returns a JSON object with allocation, predicted risk, ranked products, and optimized basket weights.

### 4. Train and evaluate

```bash
# Train the final allocation model (SupCon enabled by default)
python code/train_allocation.py --prefix allocation_best

# Compare the trained Source Encoder with practical baselines
python code/evaluate_baselines.py
```

Training uses the processed PyTorch tensors under `dataset/processed/`; to reproduce results, use the accompanying prepared data and matching model checkpoints/configuration. Additional experimental scripts in `code/` support grid search, catalog creation, case-study exports, and backtesting.

---

## Repository Structure

```text
life-to-fin-recsys/
├── checkpoints/                 # Model checkpoints and saved evaluation outputs
├── code/
│   ├── recsys/
│   │   ├── graph_builder.py     # Product similarity and score diffusion
│   │   ├── naverpay_catalog.py  # Naver Pay product loading
│   │   ├── pykrx_catalog.py     # ETF catalog loading
│   │   ├── ranker.py            # Candidate filtering and scoring
│   │   └── optimizer.py         # Allocation-aware basket weights
│   ├── models.py                # Source and Target Encoders
│   ├── train_allocation.py      # Main model training
│   ├── profile_knn.py           # Non-financial profile-space smoothing
│   ├── evaluate_baselines.py    # Allocation baseline comparison
│   ├── diversification_metrics.py
│   ├── evaluate_proxy_etf_sharpe.py
│   ├── run_end_to_end.py        # End-to-end CLI inference
│   └── run_batch_end_to_end.py  # Batch inference
├── dataset/
│   ├── catalogs/                # Product catalog snapshots
│   ├── processed/               # Encoded tensors / processing artifacts
│   ├── train.csv
│   └── test.csv
├── demo/
│   ├── app.py                   # Streamlit interface
│   └── requirements_for_demo.txt
├── requirements.txt
└── README.md
```

## Limitations & Future Work

1. **Data imbalance:** bond holdings are rare in the training data, leading to persistently small predicted bond allocations.
2. **Conservative allocation bias:** limited separation between demographic profiles across risk levels can produce similarly conservative portfolios.
3. **Sparse source signals:** 23 categorical columns cannot fully represent income uncertainty, financial goals, liquidity needs, or individual investment preferences.
4. **Limited explanations:** the current rule-based report is descriptive; the neural model's feature-to-allocation relationship remains difficult to explain causally.
5. **Market and recommendation validation:** training labels come from **U.S. SCF households**, while the example product catalog covers **Korean financial products**. Suitability for Korean users and the quality of item-level recommendations require additional validation; no user–product interaction dataset is used for collaborative filtering.

Potential extensions include richer user features, calibrated risk and asset-allocation prediction, stronger product suitability constraints, feature-attribution analysis, and user-centered evaluation of the recommended baskets.

---

## References & Acknowledgments

- [Survey of Consumer Finances (Federal Reserve)](https://www.federalreserve.gov/econres/scfindex.htm) — household demographic and financial-asset data.
- [Naver Pay Savings](https://pay.naver.com/savings/list/parking) — reference source for deposit and savings product information.
- [PyKrx](https://github.com/sharebook-kr/pykrx) — Korean securities and ETF data access.
- [Final Project Presentation](https://drive.google.com/file/d/1wzlaiiQp36cVmvfCpn4BgE2KRTilr4E1/view) — methodology, case study, evaluation figures, and limitations.

**Developed by Jungyeon Lee and Seonyeong Jeong**  
Ewha Womans University · Recommendation System Project

> **Disclaimer:** For educational and research purposes only. This repository does not provide personalized financial advice, verify current product terms, or guarantee returns or principal protection.
