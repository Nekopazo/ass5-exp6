# Current Project Model Architecture

This document tracks the current model in `/scratch/yangximing/code/ass5-exp6/DiffuFont`.

Current implementation files:

- `models/source_part_ref_dit.py`
- `models/diffusion_transformer_backbone.py`
- `models/model.py`
- `train.py`

The active generator is `SourcePartRefDiT`: a grayscale glyph x-pred diffusion model with external content-style fusion and a pure DiT backbone. The model no longer uses a U-Net refiner or block-internal style querying.

## 1. Active Model

Default architecture:

| item | value |
| --- | --- |
| model | `SourcePartRefDiT` |
| image channels | `1` |
| image size | `128x128` |
| patch size | `8` |
| patch grid | `16x16` |
| token count | `256` |
| encoder hidden dim | `256` |
| DiT hidden dim | `256` |
| DiT depth | runtime configured, commonly `12` |
| DiT heads | `8` |
| DiT MLP | `SwiGLU`, ratio `4.0` |
| normalization | `RMSNorm` |
| attention stabilization | per-head q/k RMS norm |
| conditioner dim | `512 = 256 content + 256 style` |
| final output | patch projection then unpatchify |

Style fusion is fixed to concat cross-attention. There is no runtime fusion-mode switch.

## 2. Forward Graph

Notation:

- `B`: batch size
- `R`: number of style references
- `N`: token count, `256`
- `D`: encoder hidden dim, `256`

Inputs:

| tensor | shape |
| --- | --- |
| `content_img` | `[B, 1, 128, 128]` |
| `style_img` | `[B, R, 1, 128, 128]` |
| `style_ref_mask` | `[B, R]` |
| `x_t_image` | `[B, 1, 128, 128]` |
| `timesteps` | `[B]` |

High-level path:

```text
content_tokens = ContentEncoder(content_img)                        # [B, N, D]

style_token_bank, token_valid_mask =
    StyleEncoder(style_img, style_ref_mask)                         # [B, R, N, D], [B, R, N]

conditioning_tokens =
    build_conditioning_tokens(content_tokens, style_token_bank)      # [B, N, 2D]

patch_tokens =
    DiffusionTransformerBackbone(x_t_image, timesteps, conditioning_tokens)
                                                                      # [B, N, D]

pred_x =
    decode_patch_tokens(patch_tokens, timesteps, conditioning_tokens) # [B, 1, 128, 128]
```

The diffusion model predicts `x`, not direct `v`. Training derives `v_pred` from `x_pred`.

## 3. Content Encoder

Implemented by `ContentEncoder`.

Shape trace:

| stage | shape |
| --- | --- |
| input | `[B, 1, 128, 128]` |
| stage 0 downsample/refine | `[B, 64, 64, 64]` |
| stage 1 downsample/refine | `[B, 128, 32, 32]` |
| stage 2 downsample/refine | `[B, 256, 16, 16]` |
| flatten | `[B, 256, 256]` |

The content tokens provide token positions for later content-style alignment.

## 4. Style Encoder

Implemented by `StyleEncoder`.

Each ref image is encoded independently, then regrouped:

| stage | shape |
| --- | --- |
| input | `[B, R, 1, 128, 128]` |
| flatten refs | `[B*R, 1, 128, 128]` |
| stage 0 downsample/refine | `[B*R, 64, 64, 64]` |
| stage 1 downsample/refine | `[B*R, 128, 32, 32]` |
| stage 2 downsample/refine | `[B*R, 256, 16, 16]` |
| flatten per ref | `[B*R, 256, 256]` |
| regroup | `[B, R, 256, 256]` |
| valid token mask | `[B, R, 256]` |

`style_ref_mask` masks invalid references. At least one style ref must remain valid per sample.

## 5. Style Fusion

All style refs are concatenated before cross-attention. The output is `conditioning_tokens = concat(content_tokens, style_context)` with shape `[B, 256, 512]`.

```text
style_tokens_concat = reshape(style_tokens, [B, R*N, D])
K, V = Wk/Wv(norm(style_tokens_concat))
style_key_valid_mask = reshape(style_ref_mask, [B, R*N])
style_context = Attention(Q_content, K_concat, V_concat, key_mask=style_key_valid_mask)
conditioning_tokens = concat(content_tokens, style_context)
```

This performs one cross-attention over a larger style token bank. Invalid/padded ref tokens are masked as attention keys, so they receive no softmax probability instead of being zero-valued tokens that still consume attention mass.

There is no per-ref post-attention averaging branch and no ref-level gate in the current model.

## 6. DiT Backbone

Implemented by `DiffusionTransformerBackbone` and `GlyphDiTBlock`.

Patch embedding:

```text
x = Conv2d(1 -> 128, kernel=8, stride=8)(x_t_image)                 # [B, 128, 16, 16]
x = Conv2d(128 -> 256, kernel=1)(x)                                 # [B, 256, 16, 16]
x = flatten + pos_embed                                             # [B, 256, 256]
```

Timestep embedding:

```text
time_cond = timestep_embedding(timesteps, 256)
time_cond = time_mlp(time_cond)
time_cond = time_cond_norm(time_cond)
```

Every active block uses adaLN-style modulation from timestep plus conditioning tokens:

```text
content_tokens, style_tokens = split(conditioning_tokens, 256)

content_hidden = content_condition_to_hidden(RMSNorm(content_tokens))
style_hidden = style_condition_to_hidden(RMSNorm(style_tokens))
time_hidden = block_time_to_hidden(time_cond).unsqueeze(1)          # [B, 1, 256]

joint_hidden = SiLU(time_hidden + content_hidden + style_hidden)    # [B, N, 256]

mods = joint_mod(joint_hidden)                                      # [B, N, 6*256]
```

The six modulation tensors are:

```text
self_attn_shift, self_attn_scale, self_attn_gate,
ffn_shift,       ffn_scale,       ffn_gate
```

Self-attention and FFN are both residual-gated.

## 7. Conditioning Cache

Inference now caches condition projections outside the denoising loop.

Cached once per sample call:

```text
conditioning_tokens
backbone_condition_hidden_cache =
    per-block(content_condition_to_hidden(norm(content))
            + style_condition_to_hidden(norm(style)))

output_condition_hidden =
    output_content_condition_to_hidden(norm(content))
  + output_style_condition_to_hidden(norm(style))
```

During every denoising step, the model still recomputes timestep-dependent modulation, but it reuses the expensive condition projections. This does not change numerical output; it only removes repeated work.

Training does not persistently cache sample batches on GPU. For visualization sampling, `sample_batch_builder()` is called each time `sample_every_steps` fires. A fixed seed selects one stable group of style refs, then inference is rerun from scratch for that sample image.

## 8. Output Head

Implemented by:

- `output_norm`
- `output_content_condition_norm`
- `output_style_condition_norm`
- `output_content_condition_to_hidden`
- `output_style_condition_to_hidden`
- `output_time_to_hidden`
- `output_mod`
- `output_proj`

Shape trace:

| stage | shape |
| --- | --- |
| DiT output tokens | `[B, 256, 256]` |
| final condition hidden | `[B, 256, 256]` |
| final adaLN modulation | `[B, 256, 2*256]` |
| patch projection | `[B, 256, 64]` |
| unpatchify | `[B, 1, 128, 128]` |

Final modulation path:

```text
content_hidden = output_content_condition_to_hidden(RMSNorm(content_tokens))
style_hidden = output_style_condition_to_hidden(RMSNorm(style_tokens))
time_hidden = output_time_to_hidden(time_cond).unsqueeze(1)

joint_hidden = SiLU(time_hidden + content_hidden + style_hidden)
shift, scale = output_mod(joint_hidden).chunk(2, dim=-1)

x = output_norm(patch_tokens)
x = x * (1 + scale) + shift
patch_pixels = output_proj(x)
```

`output_mod` and `output_proj` are zero-initialized for stable startup.

## 9. Training

Implemented by `XPredTrainer`.

Core x-pred training path:

```text
t = sigmoid(randn * p_std + p_mean)
x0 = noise_scale * randn
xt = t * x1 + (1 - t) * x0

pred_x = model.predict_x(xt, t, conditioning_tokens)

target_v = (x1 - xt) / clamp_min(1 - t, t_eps)
pred_v   = (pred_x - xt) / clamp_min(1 - t, t_eps)
loss = mse(pred_v, target_v)
```

Default diffusion settings:

| item | value |
| --- | --- |
| optimizer | `AdamW` |
| betas | `(0.9, 0.95)` |
| weight decay | `0.0` |
| LR schedule | warmup then constant by default, cosine optional |
| EMA | `0.9999`, starts at step `40000` |
| timestep sampling | `sigmoid(N(p_mean=-0.8, p_std=0.8))` |
| `t_eps` | `0.05` |
| noise scale | `1.0` |
| default sample steps | `20` |

## 10. Sampling

Inference uses an ODE update:

```text
for intermediate steps:
    x_euler = x_t + dt * v_pred(x_t, t)
    x_next = x_t + dt * 0.5 * (v_pred(x_t, t) + v_pred(x_euler, t + dt))

last step:
    x_next = x_t + dt * v_pred(x_t, t)
```

`v_pred` is derived from the model's `x_pred`:

```text
v_pred = (pred_x - x_t) / clamp_min(1 - t, t_eps)
```

Sample visualization behavior:

- target/style refs exclude the current target character.
- style ref candidates come from the 1000-character pool.
- training can sample a variable ref count between `style_ref_count_min` and `style_ref_count_max`.
- validation and sample visualization use `style_ref_count_max`.
- sample visualization uses the same fixed max-ref group across calls for a given seed.
- the sample batch is rebuilt every visualization call and is not kept as a long-lived GPU cache.

## 11. Checkpoint Config

Current checkpoints store `model_config` from `SourcePartRefDiT.export_config()`, including encoder/backbone dimensions, patch size, and image size. Removed fusion-mode and ref-aggregation fields are no longer part of the current model config.
