# Checkpoints

Current checkpoint prefix: `allocation_best`

This is the final SupCon-enabled checkpoint:

- `loss_weight_supcon`: `0.03`
- `embed_dim`: `32`
- `output_dim`: `512`
- `dropout`: `0.3`
- `batch_size`: `512`

Use these files for training/evaluation/inference:

- `allocation_best_checkpoint_meta.json`
- `allocation_best_source_encoder.pth`
- `allocation_best_target_encoder.pth`
- `allocation_best_training_state.pth`
- `allocation_best_metrics.json`

Use `checkpoints/baseline_comparison_supcon/` for final evaluation and results analysis.

Legacy/local-only checkpoints are ignored by git:

- `best_*` old contrastive checkpoint from 2026-05-04
- `smoke_*`, `batch_smoke_*`, `tmp_*`
- `profile_to_portfolio_smoke_*`
- `unused_*`, `obsolete_*`
- `*_test_report.json`
