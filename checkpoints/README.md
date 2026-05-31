# Checkpoints

Current checkpoint prefix: `supcon_best`

This is the final SupCon-enabled checkpoint:

- `loss_weight_supcon`: `0.03`
- `embed_dim`: `32`
- `output_dim`: `512`
- `dropout`: `0.3`
- `batch_size`: `512`

Use these files for training/evaluation/inference:

- `supcon_best_checkpoint_meta.json`
- `supcon_best_source_encoder.pth`
- `supcon_best_target_encoder.pth`
- `supcon_best_training_state.pth`
- `supcon_best_metrics.json`

`allocation_best_*` is the source checkpoint set promoted into `supcon_best_*`.
`grid_search/supcon_final_*` is the same checkpoint set copied into the grid-search checkpoint directory so `evaluate_baselines.py` can compare it with existing tuned/SupCon variants in one run.

Use `checkpoints/baseline_comparison_supcon/` for final evaluation and results analysis.

Legacy/local-only checkpoints are ignored by git:

- `best_*` old contrastive checkpoint from 2026-05-04
- `smoke_*`, `batch_smoke_*`, `tmp_*`
- `profile_to_portfolio_smoke_*`
- `unused_*`, `obsolete_*`
- `*_test_report.json`
