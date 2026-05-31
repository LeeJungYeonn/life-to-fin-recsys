# life-to-fin-recsys
A portfolio recommendation system based on demographic features

## Final Pipeline

Use these files for the current SupCon pipeline:

- `code/train_allocation.py`: trains the final source/target encoder setup with SupCon enabled by default.
- `code/grid_search_source_encoder.py`: tunes the final SupCon source encoder pipeline.
- `code/evaluate_baselines.py`: compares the trained source encoder against practical baselines.
- `code/run_end_to_end.py`: runs inference/recommendation with `allocation_best`.

Current selected checkpoint: `checkpoints/allocation_best_*`

Current result table: `checkpoints/baseline_comparison_supcon/baseline_comparison.csv`

Example commands:

```powershell
python code/train_allocation.py
python code/evaluate_baselines.py --source-prefixes supcon_003=allocation_best --output-dir checkpoints/baseline_comparison_supcon
```

The target encoder is an alignment anchor during contrastive/SupCon training, not a final baseline. Report results as source encoder inference against practical baselines:

- mean allocation
- demographic group mean
- CatBoost
- source encoder
- source encoder + KNN smoothing

Do not use source-vs-target encoder metrics as the main result table; those are only alignment diagnostics.
