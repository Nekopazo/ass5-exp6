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
- content encoder output: `16x16x256`
- style encoder output: `16x16x256`
- style fusion: one-shot `content <- style` cross-attention outside the backbone
- DiT hidden size: `256`
- DiT depth: `12`
- DiT heads: `8`
- DiT feed-forward: `swiglu`
- DiT normalization/modulation: `rms` + `shift/scale/gate`
- output path: `DiT tokens -> patch projection -> unpatchify`

Effective path:

1. Encode `content_img` into `16x16x256` content tokens.
2. Encode all style references into `16x16x256` style tokens.
3. Concatenate all style-reference tokens along the token axis.
4. Run one cross-attention with content tokens as query and style tokens as key/value.
5. Fuse `content` and `content + style_context` with `concat -> LayerNorm -> Linear(512 -> 256)`.
6. Patchify `x_t` into `256` grayscale patches and project them into the DiT hidden dimension.
7. Run the DiT backbone with timestep conditioning plus fused content tokens at every block.
8. Project final DiT tokens back to patch pixels and unpatchify to `[B, 1, 128, 128]`.

There is no U-Net refiner in the active architecture.

## 2. Main Configuration

| item | value |
| --- | --- |
| `in_channels` | `1` |
| `image_size` | `128` |
| `patch_size` | `8` |
| `patch_grid_size` | `16` |
| `num_patches` | `256` |
| `encoder_hidden_dim` | `256` |
| `dit_hidden_dim` | `256` |
| `dit_depth` | runtime-configured, default `12` |
| `dit_heads` | `8` |
| `dit_mlp_ratio` | `4.0` |
| `ffn_activation` | `swiglu` |
| `norm_variant` | `rms` |
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

pred_x = decode_patch_tokens(patch_tokens)                          # [B, 1, 128, 128]
```

## 4. Content Path

Implemented by `ContentEncoder` in `models/source_part_ref_dit.py`.

Shape trace:

| stage | shape |
| --- | --- |
| input | `[B, 1, 128, 128]` |
| stem conv | `[B, 64, 64, 64]` |
| stem block | `[B, 64, 64, 64]` |
| downsample stage 1 | `[B, 128, 32, 32]` |
| residual block 1 | `[B, 128, 32, 32]` |
| downsample stage 2 | `[B, 256, 16, 16]` |
| residual block 2 | `[B, 256, 16, 16]` |
| flatten to tokens | `[B, 256, 256]` |

## 5. Style Path

Implemented by `StyleEncoder` and `ContentStyleCrossAttention` in `models/source_part_ref_dit.py`.

Each style reference is encoded independently, then regrouped per sample.

Shape trace:

| stage | shape |
| --- | --- |
| `style_img` | `[B, R, 1, 128, 128]` |
| flatten refs | `[B*R, 1, 128, 128]` |
| stem conv | `[B*R, 64, 64, 64]` |
| stem block | `[B*R, 64, 64, 64]` |
| downsample stage 1 | `[B*R, 128, 32, 32]` |
| residual block 1 | `[B*R, 128, 32, 32]` |
| downsample stage 2 | `[B*R, 256, 16, 16]` |
| residual block 2 | `[B*R, 256, 16, 16]` |
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
| `content + style_context` | `[B, 256, 256]` |
| `conditioning_tokens = concat(...)` | `[B, 256, 512]` |

Equation:

```text
style_context = CrossAttention(Q=content_tokens, K=style_tokens, V=style_tokens)
fused_context = content_tokens + style_context
conditioning_tokens = concat([content_tokens, fused_context], dim=-1)
```

## 7. DiT Backbone

Implemented by `DiffusionTransformerBackbone` and `GlyphDiTBlock` in `models/diffusion_transformer_backbone.py`.

### 7.1 Patch and Timestep Embedding

```text
patch_pixels = patchify(x_t_image)                                  # [B, 256, 64]
x = patch_embed(patch_pixels)                                       # [B, 256, 256]
x = x + pos_embed                                                   # [B, 256, 256]

time_cond = timestep_embedding(timesteps, 256)                      # [B, 256]
time_cond = time_mlp(time_cond)                                     # [B, 256]
time_cond = time_cond_norm(time_cond)                               # [B, 256]
```

For the default `8x8` grayscale patch, `patch_dim = 64`, so the embedding MLP is:

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
time_hidden = time_to_hidden(time_cond)                            # [B, 256]
time_hidden = time_hidden.unsqueeze(1).expand(B, N, 256)          # [B, 256, 256]

cond_hidden = cond_to_hidden(norm(conditioning_tokens))            # [B, 256, 256]
joint_hidden = SiLU(time_hidden + cond_hidden)                     # [B, 256, 256]

mods = joint_mod(joint_hidden)                                     # [B, 256, 1536]
```

`time_to_hidden` is shared by the whole backbone, so every block receives the same per-sample time hidden state and only learns its own conditioner projection plus modulation head.

## 8. Output Head

Implemented by `output_norm` and `output_proj` in `models/source_part_ref_dit.py`.

Shape trace:

| stage | shape |
| --- | --- |
| DiT output tokens | `[B, 256, 256]` |
| output norm | `[B, 256, 256]` |
| patch projection | `[B, 256, 64]` |
| patch grid | `[B, 16, 16, 1, 8, 8]` |
| unpatchify | `[B, 1, 128, 128]` |

The output projection is zero-initialized so the model starts from a stable near-zero `x_pred`.

## 9. Training Path

`XPredTrainer` optimizes an `x_pred` backbone under a derived `v_pred` supervision target.

Active losses:

- v-space MSE term derived from `x_pred`

The trainer consumes the final pixel prediction and converts it to `v_pred` for the main loss:

```text
pred_x = model.predict_x(xt, timesteps, conditioning_tokens=...)
pred_v = (pred_x - xt) / (1 - t)
```

So the refactor removes the old decoder and changes the trainer contract from direct `v_pred` output to `x_pred -> v_pred`.

Inference sampling uses a fixed Euler ODE solver over `sample_steps` (default `20`), updating `x_t` with a single `v_pred` evaluation per step:

```text
x_{t + dt} = x_t + dt * v_pred(x_t, t)
```

## 10. Compatibility Note

This refactor intentionally removed:

- patch/image U-Net refiners
- `refiner_mode`
- `detailer_*` configuration
- the RMS `scale/gate` shortcut path

As a result, checkpoints produced before this refactor are not expected to load into the new model definition.
