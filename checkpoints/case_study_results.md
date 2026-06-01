# Case Study Results

- input_csv: `C:\Users\이정연\OneDrive\바탕 화면\life-to-fin-recsys\dataset\test.csv`
- checkpoint_prefix: `allocation_best`
- risk_source: `model`
- knn_smoothing: `False`
- top_k: `5`

## CASEID 165546 - Risk Label 1

| Item | Value |
| --- | --- |
| true risk label | 1 |
| predicted risk label | 1 |
| risky share | 0.71% |

### Input: non-financial categorical features

| Feature | Value |
| --- | ---: |
| OCCAT1 | 2 |
| OCCAT2 | 3 |
| INDCAT | 2 |
| LF | 1 |
| HOUSECL | 2 |
| EDCL | 1 |
| EDUC | 5 |
| AGECL | 5 |
| LIFECL | 5 |
| FAMSTRUCT | 3 |
| KIDS | 0 |
| MARRIED | 2 |
| EXPENSHILO | 2 |
| WSAVED | 2 |
| SAVRES1 | 0 |
| SAVRES2 | 0 |
| SAVRES3 | 0 |
| SAVRES4 | 0 |
| SAVRES5 | 0 |
| SAVRES6 | 0 |
| SAVRES7 | 1 |
| SAVRES8 | 0 |
| SAVRES9 | 0 |

### Allocation

| Bucket | True | Predicted |
| --- | ---: | ---: |
| cash | 100.00% | 100.00% |
| bond | 0.00% | 0.00% |
| pension | 0.00% | 0.00% |
| equity | 0.00% | 0.00% |

### Recommended Products

| Rank | Product | Bucket | Weight | Category | Type | Score |
| ---: | --- | --- | ---: | --- | --- | ---: |
| 1 | JB 슈퍼씨드 적금 | cash | 33.88% | saving | saving | 1.8685 |
| 2 | IBK부가세모으기통장 | cash | 33.07% | parking | parking | 1.8236 |
| 3 | e-그린세이브예금 | cash | 33.05% | deposit | deposit | 1.8228 |
| 4 | TIGER TDF2045 적격 | pension | 0.00% | 3 | etf | -0.2418 |
| 5 | KODEX 200 | equity | 0.00% | 5 | etf | -0.1817 |

## CASEID 176876 - Risk Label 2

| Item | Value |
| --- | --- |
| true risk label | 2 |
| predicted risk label | 2 |
| risky share | 24.45% |

### Input: non-financial categorical features

| Feature | Value |
| --- | ---: |
| OCCAT1 | 1 |
| OCCAT2 | 2 |
| INDCAT | 2 |
| LF | 1 |
| HOUSECL | 1 |
| EDCL | 3 |
| EDUC | 9 |
| AGECL | 4 |
| LIFECL | 5 |
| FAMSTRUCT | 1 |
| KIDS | 1 |
| MARRIED | 2 |
| EXPENSHILO | 3 |
| WSAVED | 2 |
| SAVRES1 | 0 |
| SAVRES2 | 0 |
| SAVRES3 | 0 |
| SAVRES4 | 1 |
| SAVRES5 | 0 |
| SAVRES6 | 0 |
| SAVRES7 | 0 |
| SAVRES8 | 0 |
| SAVRES9 | 0 |

### Allocation

| Bucket | True | Predicted |
| --- | ---: | ---: |
| cash | 51.65% | 51.71% |
| bond | 0.00% | 0.01% |
| pension | 48.35% | 48.27% |
| equity | 0.00% | 0.01% |

### Recommended Products

| Rank | Product | Bucket | Weight | Category | Type | Score |
| ---: | --- | --- | ---: | --- | --- | ---: |
| 1 | JB 슈퍼씨드 적금 | cash | 26.28% | saving | saving | 1.3857 |
| 2 | IBK부가세모으기통장 | cash | 25.43% | parking | parking | 1.3407 |
| 3 | KODEX 200미국채혼합 | bond | 0.01% | 3 | etf | -0.2173 |
| 4 | TIGER TDF2045 적격 | pension | 24.15% | 3 | etf | 0.2409 |
| 5 | KODEX TDF2050액티브 적격 | pension | 24.13% | 3 | etf | 0.2406 |

## CASEID 190931 - Risk Label 3

| Item | Value |
| --- | --- |
| true risk label | 3 |
| predicted risk label | 2 |
| risky share | 32.70% |

### Input: non-financial categorical features

| Feature | Value |
| --- | ---: |
| OCCAT1 | 1 |
| OCCAT2 | 2 |
| INDCAT | 2 |
| LF | 1 |
| HOUSECL | 1 |
| EDCL | 4 |
| EDUC | 12 |
| AGECL | 1 |
| LIFECL | 3 |
| FAMSTRUCT | 4 |
| KIDS | 2 |
| MARRIED | 1 |
| EXPENSHILO | 3 |
| WSAVED | 3 |
| SAVRES1 | 0 |
| SAVRES2 | 0 |
| SAVRES3 | 0 |
| SAVRES4 | 0 |
| SAVRES5 | 0 |
| SAVRES6 | 0 |
| SAVRES7 | 1 |
| SAVRES8 | 0 |
| SAVRES9 | 0 |

### Allocation

| Bucket | True | Predicted |
| --- | ---: | ---: |
| cash | 39.75% | 40.04% |
| bond | 0.00% | 0.02% |
| pension | 60.25% | 59.07% |
| equity | 0.00% | 0.87% |

### Recommended Products

| Rank | Product | Bucket | Weight | Category | Type | Score |
| ---: | --- | --- | ---: | --- | --- | ---: |
| 1 | JB 슈퍼씨드 적금 | cash | 20.39% | saving | saving | 1.2690 |
| 2 | IBK부가세모으기통장 | cash | 19.66% | parking | parking | 1.2240 |
| 3 | TIGER TDF2045 적격 | pension | 29.55% | 3 | etf | 0.3489 |
| 4 | KODEX TDF2050액티브 적격 | pension | 29.53% | 3 | etf | 0.3486 |
| 5 | KODEX 200 | equity | 0.87% | 5 | etf | -0.1730 |

## CASEID 236386 - Risk Label 4

| Item | Value |
| --- | --- |
| true risk label | 4 |
| predicted risk label | 3 |
| risky share | 36.70% |

### Input: non-financial categorical features

| Feature | Value |
| --- | ---: |
| OCCAT1 | 1 |
| OCCAT2 | 2 |
| INDCAT | 2 |
| LF | 1 |
| HOUSECL | 1 |
| EDCL | 4 |
| EDUC | 12 |
| AGECL | 2 |
| LIFECL | 3 |
| FAMSTRUCT | 4 |
| KIDS | 3 |
| MARRIED | 1 |
| EXPENSHILO | 3 |
| WSAVED | 2 |
| SAVRES1 | 0 |
| SAVRES2 | 1 |
| SAVRES3 | 0 |
| SAVRES4 | 0 |
| SAVRES5 | 0 |
| SAVRES6 | 0 |
| SAVRES7 | 0 |
| SAVRES8 | 0 |
| SAVRES9 | 0 |

### Allocation

| Bucket | True | Predicted |
| --- | ---: | ---: |
| cash | 31.38% | 29.40% |
| bond | 0.00% | 0.02% |
| pension | 68.62% | 69.93% |
| equity | 0.00% | 0.65% |

### Recommended Products

| Rank | Product | Bucket | Weight | Category | Type | Score |
| ---: | --- | --- | ---: | --- | --- | ---: |
| 1 | JB 슈퍼씨드 적금 | cash | 29.41% | saving | saving | 1.1626 |
| 2 | TIGER TDF2045 적격 | pension | 23.33% | 3 | etf | 0.4575 |
| 3 | KODEX TDF2050액티브 적격 | pension | 23.31% | 3 | etf | 0.4572 |
| 4 | RISE TDF2050액티브 적격 | pension | 23.31% | 3 | etf | 0.4571 |
| 5 | KODEX 200 | equity | 0.65% | 5 | etf | -0.1752 |

## CASEID 222331 - Risk Label 5

| Item | Value |
| --- | --- |
| true risk label | 5 |
| predicted risk label | 3 |
| risky share | 39.79% |

### Input: non-financial categorical features

| Feature | Value |
| --- | ---: |
| OCCAT1 | 1 |
| OCCAT2 | 3 |
| INDCAT | 1 |
| LF | 1 |
| HOUSECL | 1 |
| EDCL | 3 |
| EDUC | 10 |
| AGECL | 2 |
| LIFECL | 2 |
| FAMSTRUCT | 5 |
| KIDS | 0 |
| MARRIED | 1 |
| EXPENSHILO | 1 |
| WSAVED | 3 |
| SAVRES1 | 0 |
| SAVRES2 | 0 |
| SAVRES3 | 0 |
| SAVRES4 | 0 |
| SAVRES5 | 0 |
| SAVRES6 | 1 |
| SAVRES7 | 0 |
| SAVRES8 | 0 |
| SAVRES9 | 0 |

### Allocation

| Bucket | True | Predicted |
| --- | ---: | ---: |
| cash | 17.21% | 19.66% |
| bond | 0.00% | 0.00% |
| pension | 82.79% | 80.31% |
| equity | 0.00% | 0.04% |

### Recommended Products

| Rank | Product | Bucket | Weight | Category | Type | Score |
| ---: | --- | --- | ---: | --- | --- | ---: |
| 1 | JB 슈퍼씨드 적금 | cash | 19.66% | saving | saving | 1.0651 |
| 2 | TIGER TDF2045 적격 | pension | 20.11% | 3 | etf | 0.5612 |
| 3 | KODEX TDF2050액티브 적격 | pension | 20.10% | 3 | etf | 0.5610 |
| 4 | RISE TDF2050액티브 적격 | pension | 20.09% | 3 | etf | 0.5608 |
| 5 | KODEX TDF2060액티브 적격 | pension | 20.04% | 3 | etf | 0.5594 |
