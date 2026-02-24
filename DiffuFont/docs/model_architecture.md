# Source-Aligned Architecture (Latest)

Current training path:

`train.py -> FontImageDataset -> SourcePartRefUNet -> source_fontdiffuser.UNet -> DiffusionTrainer`

## Conditioning Profiles

Use `--conditioning-profile` in `train.py`:

- `baseline`:
  - parts_vector conditioning: OFF
  - RSI conditioning: OFF
  - behavior: only content/source path is used
- `parts_vector_only`:
  - parts_vector conditioning: ON
  - RSI conditioning: OFF
- `rsi_only`:
  - parts_vector conditioning: OFF
  - RSI conditioning: ON
- `full` (default):
  - parts_vector conditioning: ON
  - RSI conditioning: ON

## Style Path Update

The old style-image encoder branch (`E_s(x_s)`) has been removed from the runtime conditioning path.

Style conditioning is now built from PartBank part sets:

1. `parts -> patch encoder -> per-part embeddings z_i`
2. `DeepSets-style masked sum + L2 norm -> part style vector g`
3. `g` is reshaped to a length-1 conditioning sequence for MCA cross-attention when parts_vector conditioning is enabled.

## RSI Path

RSI uses style-structure features derived from `style_img` via `ContentEncoder`.
When `rsi_only` or `full` is enabled:

- `style_img` passes through `ContentEncoder`
- resulting style-structure features are used by up-block offset/deform logic

When RSI is disabled, up-blocks skip offset/deform and continue with standard skip concat.

## Losses

Main training loss in `DiffusionTrainer`:

- `loss_mse`: reconstruction MSE
- `loss_off`: RSI offset regularization from source path
- `loss_cp`: perceptual loss (VGG19)
- `loss_cons` (optional): part-vector consistency loss between two part subsets from same font

Total:

`L = lambda_mse * loss_mse + lambda_off * loss_off + lambda_cp * loss_cp + lambda_cons * loss_cons`

`lambda_cons > 0` is only valid when parts_vector conditioning is enabled; otherwise training raises an error.

## Logging / Saving Defaults

Current defaults:

- sample grid every `300` global steps
- detailed step log every `100` global steps (console + JSONL)
- checkpoint every `5000` global steps
- no epoch-based checkpoint by default

Files under `--save-dir`:

- `train_run_config.json`: full startup hyperparameter/config dump
- `train_step_metrics.jsonl`: step-level metrics log
- `ckpt_step_*.pt`: step checkpoints
- `samples/sample_ep*_gstep*_estep*.png`: sampling grids
