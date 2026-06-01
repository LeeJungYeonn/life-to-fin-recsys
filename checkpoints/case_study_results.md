# Case Study Results

- input_csv: `C:\Users\이정연\OneDrive\바탕 화면\life-to-fin-recsys\dataset\test.csv`
- checkpoint_prefix: `allocation_best`
- sample_by: `predicted-risk`
- selected_case_ids: `[165546, 176876, 163706, 201716]`
- risk_source: `model`
- knn_smoothing: `False`
- top_k: `5`

## Missing Predicted Risk Labels

No test cases were found for displayed risk label(s): 5

## CASEID 165546 - Predicted Risk Label 1

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

## CASEID 176876 - Predicted Risk Label 2

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

## CASEID 163706 - Predicted Risk Label 3

| Item | Value |
| --- | --- |
| true risk label | 3 |
| predicted risk label | 3 |
| risky share | 44.22% |

### Input: non-financial categorical features

| Feature | Value |
| --- | ---: |
| OCCAT1 | 1 |
| OCCAT2 | 1 |
| INDCAT | 2 |
| LF | 1 |
| HOUSECL | 1 |
| EDCL | 4 |
| EDUC | 12 |
| AGECL | 2 |
| LIFECL | 1 |
| FAMSTRUCT | 2 |
| KIDS | 0 |
| MARRIED | 2 |
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
| cash | 25.65% | 24.34% |
| bond | 0.00% | 0.01% |
| pension | 74.35% | 74.97% |
| equity | 0.00% | 0.68% |

### Recommended Products

| Rank | Product | Bucket | Weight | Category | Type | Score |
| ---: | --- | --- | ---: | --- | --- | ---: |
| 1 | JB 슈퍼씨드 적금 | cash | 24.35% | saving | saving | 1.1120 |
| 2 | TIGER TDF2045 적격 | pension | 25.00% | 3 | etf | 0.5078 |
| 3 | KODEX TDF2050액티브 적격 | pension | 24.99% | 3 | etf | 0.5076 |
| 4 | RISE TDF2050액티브 적격 | pension | 24.98% | 3 | etf | 0.5074 |
| 5 | KODEX 200 | equity | 0.68% | 5 | etf | -0.1749 |

## CASEID 201716 - Predicted Risk Label 4

| Item | Value |
| --- | --- |
| true risk label | 5 |
| predicted risk label | 4 |
| risky share | 55.11% |

### Input: non-financial categorical features

| Feature | Value |
| --- | ---: |
| OCCAT1 | 2 |
| OCCAT2 | 1 |
| INDCAT | 1 |
| LF | 1 |
| HOUSECL | 1 |
| EDCL | 4 |
| EDUC | 13 |
| AGECL | 4 |
| LIFECL | 5 |
| FAMSTRUCT | 5 |
| KIDS | 0 |
| MARRIED | 1 |
| EXPENSHILO | 1 |
| WSAVED | 3 |
| SAVRES1 | 0 |
| SAVRES2 | 0 |
| SAVRES3 | 1 |
| SAVRES4 | 0 |
| SAVRES5 | 0 |
| SAVRES6 | 0 |
| SAVRES7 | 0 |
| SAVRES8 | 0 |
| SAVRES9 | 0 |

### Allocation

| Bucket | True | Predicted |
| --- | ---: | ---: |
| cash | 7.70% | 39.71% |
| bond | 0.00% | 0.21% |
| pension | 7.59% | 13.57% |
| equity | 84.71% | 46.51% |

### Recommended Products

| Rank | Product | Bucket | Weight | Category | Type | Score |
| ---: | --- | --- | ---: | --- | --- | ---: |
| 1 | JB 슈퍼씨드 적금 | cash | 39.71% | saving | saving | 1.0507 |
| 2 | KODEX 200미국채혼합 | bond | 0.21% | 3 | etf | -0.2153 |
| 3 | TIGER TDF2045 적격 | pension | 13.57% | 3 | etf | -0.1061 |
| 4 | KODEX 200 | equity | 23.26% | 5 | etf | 0.2835 |
| 5 | TIGER 200 IT | equity | 23.26% | 5 | etf | 0.2835 |
