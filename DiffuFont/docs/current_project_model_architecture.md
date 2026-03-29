# Current Project Model Architecture

Generated on `2026-03-28` from the current repository state in `/scratch/yangximing/code/ass5-exp6/DiffuFont`.

This document describes:

- the current main glyph flow model
- the current grayscale CNN font perceptor
- the actual tensor flow and tensor shapes
- the current loss composition
- the remaining redundancy candidates in the codebase

The shape traces below were obtained with:

```bash
/scratch/yangximing/miniconda3/envs/sg3/bin/python scripts/inspect_flow_model.py --device cpu --batch-size 2 --style-refs 6
/scratch/yangximing/miniconda3/envs/sg3/bin/python scripts/inspect_font_perceptor.py --device cpu --batch-size 2 --image-size 128
```

## 1. Project Overview

The current project has two active model paths:

1. `SourcePartRefDiT`
   - used by the main pixel-space glyph flow training
   - input is grayscale glyph image space, not latent space
   - content is injected only in the first 6 DiT blocks
   - style is injected only in the last 6 DiT blocks
   - style conditioning is now pure AdaLN-style modulation
   - there is no style cross-attention anymore

2. `FontPerceptor`
   - used for grayscale CNN perceptual/style supervision
   - can be pretrained independently
   - can be loaded into main training as a frozen guidance network

## 2. Main Model: SourcePartRefDiT

### 2.1 Default active configuration

Current default structure resolved from the model:

| item | value |
| --- | --- |
| `in_channels` | `1` |
| `image_size` | `128` |
| `patch_size` | `16` |
| `patch_grid_size` | `8` |
| `num_patches` | `64` |
| `encoder_hidden_dim` | `512` |
| `style_hidden_dim` | `512` |
| `dit_hidden_dim` | `512` |
| `dit_depth` | `12` |
| `dit_heads` | `8` |
| `content_fusion_layers` | `[0, 6)` |
| `style_fusion_layers` | `[6, 12)` |
| `detailer_bottleneck_channels` | `384` |

### 2.2 Top-level module composition

Measured parameter counts:

| module | params |
| --- | ---: |
| `content_encoder` | `13,726,336` |
| `content_proj` | `0` |
| `style_encoder` | `11,953,920` |
| `style_global_proj` | `0` |
| `backbone` | `60,784,640` |
| `detailer` | `4,852,737` |
| total | `91,317,633` |

### 2.3 Backbone block plan

The 12 DiT blocks are split exactly as follows:

| block | content injection | style modulation |
| --- | --- | --- |
| `0` | yes | no |
| `1` | yes | no |
| `2` | yes | no |
| `3` | yes | no |
| `4` | yes | no |
| `5` | yes | no |
| `6` | no | yes |
| `7` | no | yes |
| `8` | no | yes |
| `9` | no | yes |
| `10` | no | yes |
| `11` | no | yes |

This means:

- front 6 layers only fuse content tokens
- back 6 layers only receive style-conditioned AdaLN modulation
- style no longer enters through cross-attention

## 3. Main Model Tensor Flow

The shape trace below uses:

- batch size `B=2`
- style reference count `R=6`
- image size `128x128`

### 3.1 Input tensors

| tensor | shape |
| --- | --- |
| `content` | `(2, 1, 128, 128)` |
| `target(x1)` | `(2, 1, 128, 128)` |
| `style_img` | `(2, 6, 1, 128, 128)` |
| `style_ref_mask` | `(2, 6)` |
| `x0_noise` | `(2, 1, 128, 128)` |
| `timesteps` | `(2,)` |
| `xt=(1-t)*x0+t*x1` | `(2, 1, 128, 128)` |
| `target_flow=x1-x0` | `(2, 1, 128, 128)` |

### 3.2 Content path

`content` is passed through `ContentEncoder`, then flattened into patch-aligned tokens:

| stage | shape |
| --- | --- |
| `content_input` | `(2, 1, 128, 128)` |
| `content.stem` | `(2, 64, 64, 64)` |
| `content.stem_resblock` | `(2, 64, 64, 64)` |
| `content.downsample_0` | `(2, 128, 32, 32)` |
| `content.resblock_0` | `(2, 128, 32, 32)` |
| `content.downsample_1` | `(2, 256, 16, 16)` |
| `content.resblock_1` | `(2, 256, 16, 16)` |
| `content.resblock_1_extra1` | `(2, 256, 16, 16)` |
| `content.downsample_2` | `(2, 512, 8, 8)` |
| `content.resblock_2` | `(2, 512, 8, 8)` |
| `content.resblock_2_extra1` | `(2, 512, 8, 8)` |
| `content.out_norm_silu` | `(2, 512, 8, 8)` |
| `content.tokens_before_proj` | `(2, 64, 512)` |
| `content.tokens_after_proj` | `(2, 64, 512)` |

Interpretation:

- content path reduces `128x128` to `8x8`
- `8x8` aligns exactly with the `16x16` patch grid of the DiT branch
- after flattening, there is one content token per image patch

### 3.3 Style path

`style_img` contains `R=6` grayscale style references per sample.

Each reference is encoded independently, pooled to a per-reference vector, then averaged with `style_ref_mask`.

| stage | shape |
| --- | --- |
| `style_input` | `(2, 6, 1, 128, 128)` |
| `style.flatten_refs` | `(12, 1, 128, 128)` |
| `style.downsample_0` | `(12, 64, 64, 64)` |
| `style.resblock_0` | `(12, 64, 64, 64)` |
| `style.downsample_1` | `(12, 128, 32, 32)` |
| `style.resblock_1` | `(12, 128, 32, 32)` |
| `style.downsample_2` | `(12, 256, 16, 16)` |
| `style.resblock_2` | `(12, 256, 16, 16)` |
| `style.downsample_3` | `(12, 384, 8, 8)` |
| `style.resblock_3` | `(12, 384, 8, 8)` |
| `style.downsample_4` | `(12, 512, 4, 4)` |
| `style.resblock_4` | `(12, 512, 4, 4)` |
| `style.global_pool_flatten` | `(12, 512)` |
| `style.per_ref_vectors` | `(2, 6, 512)` |
| `style.pooled_style` | `(2, 512)` |
| `style.global` | `(2, 512)` |

Interpretation:

- style path collapses all spatial information into a single `style_global` vector per sample
- the DiT backbone only consumes `style_global`

### 3.4 DiT backbone path

The noisy glyph `xt` is patchified by a strided conv and processed as a 64-token sequence.

| stage | shape |
| --- | --- |
| `xt_input` | `(2, 1, 128, 128)` |
| `backbone.patch_embed_tokens` | `(2, 64, 512)` |
| `backbone.tokens_plus_pos` | `(2, 64, 512)` |
| `backbone.timestep_embedding` | `(2, 512)` |
| `backbone.time_mlp` | `(2, 512)` |
| `backbone.style_cond_proj` | `(2, 512)` |
| `backbone.block_00` to `block_11` | always `(2, 64, 512)` |
| `backbone.final_norm` | `(2, 64, 512)` |

Block semantics:

- blocks `0..5`
  - use content token fusion
  - do not use style modulation
- blocks `6..11`
  - do not use content fusion
  - add `style_cond` into the AdaLN conditioning path

### 3.5 Patch detailer path

The backbone outputs one token per patch. The detailer refines each `16x16` patch with a shallow local U-Net made of `Conv+SiLU+pool` down blocks, bottleneck context concatenation, `Upsample+Conv+SiLU` up blocks, and a final `1x1` output convolution.

| stage | shape |
| --- | --- |
| `model._patchify(xt)` | `(2, 64, 1, 16, 16)` |
| `detailer.flat_patches` | `(128, 1, 16, 16)` |
| `detailer.flat_tokens` | `(128, 512)` |
| `detailer.enc_block_0` | `(128, 32, 16, 16)` |
| `detailer.downsample_0` | `(128, 32, 8, 8)` |
| `detailer.enc_block_1` | `(128, 64, 8, 8)` |
| `detailer.downsample_1` | `(128, 64, 4, 4)` |
| `detailer.enc_block_2` | `(128, 128, 4, 4)` |
| `detailer.downsample_2` | `(128, 128, 2, 2)` |
| `detailer.enc_block_3` | `(128, 256, 2, 2)` |
| `detailer.downsample_3` | `(128, 256, 1, 1)` |
| `detailer.context_proj` | `(128, 384, 1, 1)` |
| `detailer.concat_context` | `(128, 640, 1, 1)` |
| `detailer.bottleneck` | `(128, 384, 1, 1)` |
| `detailer.upsample_0` | `(128, 384, 2, 2)` |
| `detailer.concat_skip_0` | `(128, 640, 2, 2)` |
| `detailer.dec_block_0` | `(128, 256, 2, 2)` |
| `detailer.upsample_1` | `(128, 256, 4, 4)` |
| `detailer.concat_skip_1` | `(128, 384, 4, 4)` |
| `detailer.dec_block_1` | `(128, 128, 4, 4)` |
| `detailer.upsample_2` | `(128, 128, 8, 8)` |
| `detailer.concat_skip_2` | `(128, 192, 8, 8)` |
| `detailer.dec_block_2` | `(128, 64, 8, 8)` |
| `detailer.upsample_3` | `(128, 64, 16, 16)` |
| `detailer.concat_skip_3` | `(128, 96, 16, 16)` |
| `detailer.dec_block_3` | `(128, 32, 16, 16)` |
| `detailer.out_proj` | `(128, 1, 16, 16)` |
| `detailer.pred_patches` | `(2, 64, 1, 16, 16)` |

### 3.6 Final outputs

| tensor | shape |
| --- | --- |
| `pred_flow` | `(2, 1, 128, 128)` |
| `pred_target=xt+(1-t)*pred_flow` | `(2, 1, 128, 128)` |

## 4. Main Training Logic

### 4.1 Forward equations

Current flow training uses:

```text
x1 = target
x0 ~ N(0, I)
t ~ Uniform(0, 1)
xt = (1 - t) * x0 + t * x1
target_flow = x1 - x0

content_tokens = ContentEncoder(content) -> flatten -> content_proj
style_global = StyleEncoder(style_refs) -> per-ref pool -> masked mean -> style_global_proj

pred_flow = SourcePartRefDiT.predict_flow(xt, t, content_tokens, style_global)
pred_target = xt + (1 - t) * pred_flow
```

### 4.2 Loss composition

The current main loss is:

```text
loss_unscaled =
    lambda_flow * loss_flow
  + perceptual_weight(step) * loss_perceptual
  + style_weight(step) * loss_style_embed

loss = total_loss_scale(step) * loss_unscaled
```

Where:

- `loss_flow`
  - MSE between `pred_flow` and `target_flow`
- `loss_perceptual`
  - sum of L1 distances over multiple perceptor feature maps
- `loss_style_embed`
  - cosine distance between perceptor style embeddings of `pred_target` and `target`

Current scheduling:

- `perceptual_weight(step)` and `style_weight(step)`
  - ramp linearly from `0` to target value in the first `cnn_ramp_steps`
  - then decay together with the last-20%-of-training schedule
- `total_loss_scale(step)`
  - warmup at the beginning
  - hold at `1.0` in the middle
  - linearly decay to `total_loss_min_scale` in the last `20%` of training

### 4.3 Sampling flow

Current inference sampling is explicit Euler integration in image space:

```text
sample_0 ~ N(0, I)
for step_idx in [0, N-1]:
    t = step_idx / N
    pred_flow = model(sample, t, content, style_refs)
    sample = sample + (1 / N) * pred_flow
```

There is no VAE and no latent decoding in the active path.

## 5. CNN Font Perceptor

### 5.1 Default active configuration

| item | value |
| --- | --- |
| `in_channels` | `1` |
| `base_channels` | `32` |
| `proj_dim` | `128` |
| `num_chars` | `1000` in the bare probe |
| `feature_stage_names` | `["stage1", "stage2", "stage3", "stage4"]` |

### 5.2 Parameter counts

| module | params |
| --- | ---: |
| `stem` | `5,344` |
| `stage1` | `25,280` |
| `stage2` | `68,096` |
| `stage3` | `238,464` |
| `stage4` | `446,848` |
| `global_proj` | `66,304` |
| `style_proj_head` | `98,688` |
| `char_head` | `322,792` |
| total | `1,271,816` |

### 5.3 Forward path and shapes

The probe below uses `B=2`, `image_size=128`.

| stage | shape |
| --- | --- |
| `input` | `(2, 1, 128, 128)` |
| `stem` | `(2, 32, 64, 64)` |
| `stage1` | `(2, 64, 32, 32)` |
| `stage2` | `(2, 128, 16, 16)` |
| `stage3` | `(2, 192, 8, 8)` |
| `stage4` | `(2, 256, 4, 4)` |
| `global_pool_flatten` | `(2, 256)` |
| `global_feat` | `(2, 256)` |
| `style_embed` | `(2, 128)` |
| `char_logits` | `(2, 1000)` |

The perceptor returns:

- `feature_maps`
  - selected from `stage1` to `stage4`
- `global_feat`
  - pooled global vector
- `style_embed`
  - normalized embedding for style similarity supervision
- `char_logits`
  - character classification logits

## 6. Perceptor Pretraining Logic

Perceptor pretraining uses:

```text
images = batch["target"]
char_ids = batch["char_id"]
font_ids = batch["font_id"]

outputs = FontPerceptor(images)
loss_char_ce = CE(outputs["char_logits"], char_ids)
loss_style_supcon = SupCon(outputs["style_embed"], font_ids)

loss = loss_char_ce + style_supcon_lambda * loss_style_supcon
```

Validation qualification is based on:

- `char_acc`
- `style_cos_margin = style_pos_cos - style_neg_cos`
- positive and negative pair counts both being non-zero

The final `qualification_report.json` decides:

- whether the perceptor is qualified
- whether it can be integrated into main flow training directly

## 7. Redundancy Check

### 7.1 Main active path

After the recent cleanups, the current active main path is internally consistent:

- no style cross-attention branch remains in the backbone
- first 6 blocks are content-only
- last 6 blocks are style-modulated only
- the flow path is fully image-space

There is no large dead branch left inside the active forward path.

### 7.2 Remaining redundancy candidates

No obvious dead branch remains in the active training or inference path.

The only stale pieces still visible in the repository are analysis artifacts, not runnable model code:

- `analysis/_tmp_vae_latent_reg_smoke/train_config.json`
- `analysis/memory_profile.json`

These files still contain older VAE-era or pre-cleanup metadata keys such as `encoder_patch_size` or `freeze_style_global`, but they are historical outputs and are not imported by the current code.

### 7.3 Recommended cleanup priority

If you want to continue trimming the repo, the recommended order is:

1. archive or delete old VAE-era analysis artifacts under `analysis/` if they are no longer needed
2. review whether helper scripts outside the main training path should also be forced into explicit-arg mode for consistency

## 8. Reproducibility

Useful inspection commands:

```bash
/scratch/yangximing/miniconda3/envs/sg3/bin/python scripts/inspect_flow_model.py --device cpu --batch-size 2 --style-refs 6
/scratch/yangximing/miniconda3/envs/sg3/bin/python scripts/inspect_font_perceptor.py --device cpu --batch-size 2 --image-size 128
```

Current architecture documentation file:

```text
docs/current_project_model_architecture.md
```
