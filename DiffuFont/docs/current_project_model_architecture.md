# Current Project Model Architecture

Generated from the current repository state in `/scratch/yangximing/code/ass5-exp6/DiffuFont`.

This document describes only the active refactored model. Historical U-Net decoder branches and backward-compatible fallbacks are intentionally removed.

## 1. Main Path

Current model:

- model: `SourcePartRefDiT`
- input/output channels: `1`
- image size: `128x128`
- default patch size: `8`
- patch lattice: `16x16 = 256` tokens
- patch embed bottleneck: `128`
- content encoder output: `16x16x256`
- style encoder output: `16x16x256`
- content encoder block depth: `2` `DBlock`s per stage
- style encoder block depth: `2` `DBlock`s per stage
- style fusion: one-shot `content <- style` cross-attention outside the backbone
- DiT hidden size: `256`
- DiT depth: `12`
- DiT heads: `8`
- DiT feed-forward: `swiglu`
- DiT normalization/modulation: `rms` + `shift/scale/gate`
- attention stabilization: head-wise `qk` RMS norm
- output path: `DiT tokens -> final adaLN -> patch projection -> unpatchify`

Effective path:

1. Encode `content_img` with a `DBlock` conv pyramid into `16x16x256` content tokens.
2. Encode all style references with a matching `DBlock` conv pyramid into `16x16x256` style tokens.
3. Concatenate all style-reference tokens along the token axis.
4. Run one cross-attention with content tokens as query and style tokens as key/value.
5. Concatenate `content` and `style_context` into a `512`-channel conditioner.
6. Run bottleneck conv patch embedding on `x_t`: `Conv2d(1 -> 128, k=8, s=8)` then `Conv2d(128 -> 256, k=1)`.
7. Run the DiT backbone with timestep conditioning plus the `512`-channel conditioner at every block.
8. Run a conditioner-aware final adaLN head, then project final DiT tokens back to patch pixels and unpatchify to `[B, 1, 128, 128]`.

There is no U-Net refiner in the active architecture.

## 2. Main Configuration

| item | value |
| --- | --- |
| `in_channels` | `1` |
| `image_size` | `128` |
| `patch_size` | `8` |
| `patch_grid_size` | `16` |
| `num_patches` | `256` |
| `patch_embed_bottleneck_dim` | `128` |
| `encoder_hidden_dim` | `256` |
| `dit_hidden_dim` | `256` |
| `dit_depth` | runtime-configured, default `12` |
| `dit_heads` | `8` |
| `dit_mlp_ratio` | `4.0` |
| `ffn_activation` | `swiglu` |
| `norm_variant` | `rms` |
| `attention_qk_norm` | `head-wise rms` |
| `content_style_fusion_heads` | `4` |
| `content_injection_layers` | all DiT layers |

## 3. Top-Level Forward Graph

Notation:

- `B`: batch size
- `R`: number of style references
- `H = W = 128`
- `P = 8`
- `N = (H / P)^2 = 256`
- `C = 256`
- `D = 256`

Inputs:

| tensor | shape |
| --- | --- |
| `content_img` | `[B, 1, 128, 128]` |
| `style_img` | `[B, R, 1, 128, 128]` |
| `style_ref_mask` | `[B, R]` |
| `x_t_image` | `[B, 1, 128, 128]` |
| `timesteps` | `[B]` |

Forward:

```text
content_tokens = encode_content_tokens(content_img)                 # [B, 256, 256]

style_token_bank, token_valid_mask =
    encode_style_token_bank(style_img, style_ref_mask)              # [B, R*256, 256], [B, R*256]

conditioning_tokens = fuse_content_style_tokens(
    content_tokens,
    style_token_bank,
    token_valid_mask=token_valid_mask,
)                                                                   # [B, 256, 512]

patch_tokens = backbone(
    x_t_image,
    timesteps,
    conditioning_tokens=conditioning_tokens,
)                                                                   # [B, 256, 256]

pred_x = decode_patch_tokens(
    patch_tokens,
    timesteps=timesteps,
    conditioning_tokens=conditioning_tokens,
)                                                                   # [B, 1, 128, 128]
```

## 4. Content Path

Implemented by `ContentEncoder` in `models/source_part_ref_dit.py`.

Shape trace:

| stage | shape |
| --- | --- |
| input | `[B, 1, 128, 128]` |
| stage 0 downsample `DBlock` | `[B, 64, 64, 64]` |
| stage 0 refine `DBlock` | `[B, 64, 64, 64]` |
| stage 1 downsample `DBlock` | `[B, 128, 32, 32]` |
| stage 1 refine `DBlock` | `[B, 128, 32, 32]` |
| stage 2 downsample `DBlock` | `[B, 256, 16, 16]` |
| stage 2 refine `DBlock` | `[B, 256, 16, 16]` |
| flatten to tokens | `[B, 256, 256]` |

## 5. Style Path

Implemented by `StyleEncoder` and `ContentStyleCrossAttention` in `models/source_part_ref_dit.py`.

Each style reference is encoded independently, then regrouped per sample.

Shape trace:

| stage | shape |
| --- | --- |
| `style_img` | `[B, R, 1, 128, 128]` |
| flatten refs | `[B*R, 1, 128, 128]` |
| stage 0 downsample `DBlock` | `[B*R, 64, 64, 64]` |
| stage 0 refine `DBlock` | `[B*R, 64, 64, 64]` |
| stage 1 downsample `DBlock` | `[B*R, 128, 32, 32]` |
| stage 1 refine `DBlock` | `[B*R, 128, 32, 32]` |
| stage 2 downsample `DBlock` | `[B*R, 256, 16, 16]` |
| stage 2 refine `DBlock` | `[B*R, 256, 16, 16]` |
| flatten per ref | `[B*R, 256, 256]` |
| regroup refs | `[B, R*256, 256]` |
| token valid mask | `[B, R*256]` |

## 6. External Content-Style Cross Attention

The style fusion happens exactly once outside the backbone.

| tensor | shape |
| --- | --- |
| query = `content_tokens` | `[B, 256, 256]` |
| key = `style_token_bank` | `[B, R*256, 256]` |
| value = `style_token_bank` | `[B, R*256, 256]` |
| `style_context` | `[B, 256, 256]` |
| `conditioning_tokens = concat(...)` | `[B, 256, 512]` |

Equation:

```text
style_context = CrossAttention(Q=content_tokens, K=style_tokens, V=style_tokens)
conditioning_tokens = concat([content_tokens, style_context], dim=-1)
```

## 7. DiT Backbone

Implemented by `DiffusionTransformerBackbone` and `GlyphDiTBlock` in `models/diffusion_transformer_backbone.py`.

### 7.1 Patch and Timestep Embedding

```text
x = Conv2d(1 -> 128, kernel=8, stride=8)(x_t_image)                # [B, 128, 16, 16]
x = Conv2d(128 -> 256, kernel=1, stride=1)(x)                      # [B, 256, 16, 16]
x = flatten(x)                                                     # [B, 256, 256]
x = x + pos_embed                                                  # [B, 256, 256]

time_cond = timestep_embedding(timesteps, 256)                     # [B, 256]
time_cond = time_mlp(time_cond)                                    # [B, 256]
time_cond = time_cond_norm(time_cond)                              # [B, 256]
```

For the default `8x8` grayscale patch, raw patch dim is `64`, and the active conv embed is:

```text
64 -> 128 -> 256
```

### 7.2 Block Conditioning

Each block receives:

- current patch tokens
- timestep conditioning
- full external conditioner tokens

Every active block predicts six modulation tensors:

```text
self_attn_shift, self_attn_scale, self_attn_gate,
ffn_shift,       ffn_scale,       ffn_gate
```

Rule:

```text
norm_x = RMSNorm(x)
modulated_x = norm_x * (1 + scale) + shift
x = x + gate * branch_out
```

There is no reduced `scale/gate` shortcut path in the active implementation.

Current conditioning path inside each block:

```text
cond_hidden = cond_to_hidden(norm(conditioning_tokens))            # [B, 256, 256]
time_hidden = block_time_to_hidden(time_cond)                      # [B, 256]
time_hidden = time_hidden.unsqueeze(1).expand(B, N, 256)          # [B, 256, 256]
joint_hidden = SiLU(time_hidden + cond_hidden)                     # [B, 256, 256]

mods = joint_mod(joint_hidden)                                     # [B, 256, 1536]
```

`time_cond` is shared by the whole backbone, but every block owns its own `time_to_hidden`, so each layer can learn its own timestep interpretation before mixing with the conditioner.

Inside each attention layer, projected `q` and `k` are additionally normalized with per-head RMS norm before flash SDPA.

## 8. Output Head

Implemented by `output_norm`, `output_condition_norm`, `output_condition_to_hidden`, `output_time_to_hidden`, `output_mod`, and `output_proj` in `models/source_part_ref_dit.py`.

Shape trace:

| stage | shape |
| --- | --- |
| DiT output tokens | `[B, 256, 256]` |
| final conditioner hidden | `[B, 256, 256]` |
| final adaLN modulation | `[B, 256, 256]` |
| patch projection | `[B, 256, 64]` |
| patch grid | `[B, 16, 16, 1, 8, 8]` |
| unpatchify | `[B, 1, 128, 128]` |

The final modulation head and output projection are zero-initialized so the model starts from a stable near-zero `x_pred`.

## 9. Training Path

`XPredTrainer` now mirrors JiT's `x_pred -> derived v_pred` training path.

Active losses:

- v-space MSE term derived from `x_pred`
- optional dual-ref random-style supervision where both ref sets learn the same target with JiT-style `v-loss`

The trainer samples timestep with a logistic-normal distribution and applies the same denominator clamp used during inference:

```text
t = sigmoid(randn * p_std + p_mean)
x0 = noise_scale * randn
xt = t * x1 + (1 - t) * x0

target_v = (x1 - xt) / clamp_min(1 - t, t_eps)
pred_x = model.predict_x(xt, timesteps, conditioning_tokens=...)
pred_v = (pred_x - xt) / clamp_min(1 - t, t_eps)
```

So the refactor removes the old decoder and changes the trainer contract from direct `v_pred` output to `x_pred -> derived v_pred`.

When `consistency_lambda > 0` and `consistency_start_step` is explicitly reached, training also samples a second random style-reference set for the same target/font pair. Both ref sets use the same `xt` and `t`, and both branches are directly supervised toward the same target with JiT-style `v-loss`; the two branch losses are averaged with `consistency_lambda` used as the alternate-ref branch weight. `pred_x_ref_diff_l1` is logged only as a diagnostic and is not part of the loss.

Inference sampling now matches JiT's ODE update more closely: Heun is used for all intermediate steps, and the last step falls back to Euler.

```text
for intermediate steps:
x_euler = x_t + dt * v_pred(x_t, t)
x_{t + dt} = x_t + dt * 0.5 * (v_pred(x_t, t) + v_pred(x_euler, t + dt))

last step:
x_{t + dt} = x_t + dt * v_pred(x_t, t)
```

Active optimizer and schedule defaults for diffusion training:

- optimizer: `AdamW`
- betas: `(0.9, 0.95)`
- weight decay: `0.0`
- learning-rate schedule: warmup then `constant` by default, optional `cosine`
- EMA: `0.9999`, enabled from step `40000`
- dual-ref random-style supervision: alternate-ref branch weight `= 1.0`, start step must be set explicitly
- timestep sampling: `t = sigmoid(N(p_mean=-0.8, p_std=0.8))`
- denominator clamp: `t_eps = 0.05`
- noise scale: `1.0`
- default sampling steps: `20`

## 10. Compatibility Note

This refactor intentionally removed:

- patch/image U-Net refiners
- `refiner_mode`
- `detailer_*` configuration
- the RMS `scale/gate` shortcut path

As a result, checkpoints produced before this refactor are not expected to load into the new model definition.
