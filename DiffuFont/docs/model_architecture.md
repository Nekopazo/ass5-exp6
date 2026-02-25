# Source-Aligned Architecture (Latest)

Current training path:

`train.py -> FontImageDataset -> SourcePartRefUNet -> source_fontdiffuser.UNet -> DiffusionTrainer`

## Core Design (Few-Part Redesign)

- Input/output path:
  - noisy target `x_t` is `256x256`
  - stem: `256 -> 128` (stride-2 conv)
  - UNet main path runs on `128 -> 64 -> 32 -> 16 -> 32 -> 64 -> 128`
  - head: `128 -> 256` (upsample + conv)
- Content path:
  - `ContentEncoder(256)` provides multi-scale features
  - down blocks use `c128/c64/c32/c16`
  - Mid block does **not** inject content (hard constraint)
- Style path:
  - style comes from online retrieved PartBank parts (not direct user input)
  - `part CNN -> per-part embedding`
  - DeepSets aggregation: masked `mean + LayerNorm`
  - MLP maps set vector to fixed tokens `S in R^(M x d)`
  - current default: `M=8`, `d=256`
- Style-attention scales:
  - default only at `32/16` plus Mid and Up(16/32)
  - controlled by `--attn-scales` (default `16,32`)

## Conditioning Profiles

Use `--conditioning-profile` in `train.py`:

- `baseline`: parts OFF, RSI OFF
- `parts_vector_only` (default): parts ON, RSI OFF
- `rsi_only`: parts OFF, RSI ON
- `full`: parts ON, RSI ON

## RSI Path

RSI uses style-structure features from `style_img` through `ContentEncoder`.

- Enabled only for `rsi_only/full`
- Up blocks `StyleRSIUpBlock2D` apply fixed `OffsetRefStrucInter + DeformConv2d`
- When disabled, offset term is zero and up blocks skip RSI deformation
- `disable_self_attn` and `lite_daca` runtime switches are removed; attention/RSI block structure is now fixed in code
- Attention placement (5 scales): `Down-32`, `Down-16`, `Mid-16`, `Up-16`, `Up-32`
- Attention order inside each enabled block is fixed: `ResBlock -> Self-Attention -> Cross-Attention(style) -> ResBlock`

## Dataset / PartBank Sampling

- Parts are sampled directly from the PartBank by font label (no retrieval CNN)
- Random 1-8 parts per sample (`--part-set-min=1`, `--part-set-max=8`)
- Two independent part sets (parts / parts_b) are drawn per sample for InfoNCE contrastive learning
- Collate pads to batch max and emits `part_mask`; per-batch `font_ids` are computed for InfoNCE

## Losses

Main training loss in `DiffusionTrainer`:

- `loss_mse`: diffusion reconstruction MSE (ε-prediction)
- `loss_off`: RSI offset regularization
- `loss_nce`: InfoNCE contrastive loss on dual part-set views (λ=0.05)

Total:

`L = lambda_mse * loss_mse + lambda_off * loss_off + lambda_nce * loss_nce`

## Runtime Defaults (train.py)

- `conditioning_profile=parts_vector_only`
- `attn_scales=16,32`
- `part_set_max=8`
- `part_set_min=1`
- `lambda_nce=0.05`
- `style_token_count=8`
- `style_token_dim=256`
