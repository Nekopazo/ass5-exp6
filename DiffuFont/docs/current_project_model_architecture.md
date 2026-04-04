# Current Project Model Architecture

Generated on `2026-04-03` from the current repository state in `/scratch/yangximing/code/ass5-exp6/DiffuFont`.

This document records the current runnable architecture after the DiT conditioning refactor:

- `SourcePartRefDiT` for pixel-space glyph flow generation
- `DiffusionTransformerBackbone` block-by-block conditioning order
- `PatchDetailerHead` local patch refinement head
- `FlowTrainer` loss composition and sampling path
- `FontPerceptor` frozen perceptual/style guidance model

Shape traces and parameter counts were checked with:

```bash
/scratch/yangximing/miniconda3/envs/sg3/bin/python scripts/inspect_flow_model.py --device cpu --batch-size 1 --style-refs 6
/scratch/yangximing/miniconda3/envs/sg3/bin/python scripts/inspect_font_perceptor.py --device cpu --batch-size 2 --image-size 128
```

## 1. Active Model Paths

### 1.1 `SourcePartRefDiT`

Used by `train.py` for grayscale pixel-space flow matching.

Core design:

- content is encoded into one token per `16x16` image patch
- style is encoded into one global vector per sample
- each transformer block applies external content/style injection first, then a standard time-conditioned DiT block
- content injection uses zero-linear residuals
- style injection uses zero-linear residuals from the global style vector
- style does not enter AdaLN
- self-attention is performed inside the DiT block after content/style injection
- patch-level transformer outputs are refined by a per-patch local detailer before unpatchifying to the final flow field

### 1.2 `FontPerceptor`

Used by `train_font_perceptor.py` for pretraining, and optionally loaded by `FlowTrainer` as a frozen guidance model.

Core design:

- grayscale CNN backbone with four downsampling stages
- global pooled style embedding head for font-style supervision
- character classification head for auxiliary character recognition
- frozen guidance computes feature-map L1 perceptual loss and style embedding cosine loss

## 2. `SourcePartRefDiT` Default Configuration

Current default constructor values:

| item | value |
| --- | --- |
| `in_channels` | `1` |
| `image_size` | `128` |
| `patch_size` | `16` |
| `patch_grid_size` | `8` |
| `num_patches` | `64` |
| `encoder_hidden_dim` | `512` |
| `style_hidden_dim` | `786` |
| `style_pool_heads` | `6` |
| `dit_hidden_dim` | `512` |
| `dit_depth` | `12` |
| `dit_heads` | `8` |
| `dit_mlp_ratio` | `4.0` |
| `content_injection_layers` | `[1, 2, 3, 4, 5, 6]` |
| `style_injection_layers` | `[7, 8, 9, 10, 11, 12]` |
| `detailer_base_channels` | `32` |
| `detailer_max_channels` | `256` |
| `detailer_bottleneck_channels` | `384` |

Current parameter counts from `inspect_flow_model.py`:

| module | params |
| --- | ---: |
| `content_encoder` | `7,823,488` |
| `content_proj` | `0` |
| `style_encoder` | `19,304,518` |
| `style_pool` | `2,475,114` |
| `style_proj` | `402,944` |
| `backbone` | `60,531,200` |
| `detailer` | `4,852,737` |
| total | `95,390,001` |

## 3. Top-Level Forward Graph

Let `B` be batch size, `R` be style reference count, `H=W=128`, `P=16`, `N=(H/P)^2=64`, `D=512`.

Inputs:

| tensor | shape |
| --- | --- |
| `content_img` | `[B, 1, 128, 128]` |
| `style_img` | `[B, R, 1, 128, 128]` |
| `style_ref_mask` | `[B, R]` |
| `x_t_image` | `[B, 1, 128, 128]` |
| `timesteps` | `[B]` |

Top-level forward equations:

```text
content_tokens = content_proj(encode_content_tokens(content_img))     # [B, 64, 512]
style_global = encode_style_global(style_img, style_ref_mask)         # [B, 512]
patch_tokens = backbone(x_t_image, timesteps,
                        content_tokens=content_tokens,
                        style_global=style_global)                    # [B, 64, 512]
noisy_patches = patchify(x_t_image)                                  # [B, 64, 1, 16, 16]
pred_patches = detailer(patch_tokens, noisy_patches)                  # [B, 64, 1, 16, 16]
pred_flow = unpatchify(pred_patches)                                  # [B, 1, 128, 128]
```

## 4. Content Encoder

Implemented by `ContentEncoder` in `models/source_part_ref_dit.py`.

Shape trace for `B=1`:

| stage | shape |
| --- | --- |
| `content_input` | `(1, 1, 128, 128)` |
| `stem` | `(1, 64, 64, 64)` |
| `stem_resblock` | `(1, 64, 64, 64)` |
| `downsample_0` | `(1, 128, 32, 32)` |
| `resblock_0` | `(1, 128, 32, 32)` |
| `downsample_1` | `(1, 256, 16, 16)` |
| `resblock_1` | `(1, 256, 16, 16)` |
| `downsample_2` | `(1, 512, 8, 8)` |
| `resblock_2` | `(1, 512, 8, 8)` |
| `out_norm + SiLU` | `(1, 512, 8, 8)` |
| `flatten(2).transpose(1,2)` | `(1, 64, 512)` |
| `content_proj` | `(1, 64, 512)` |

Semantics:

- the encoder downsamples content glyphs to the same `8x8` grid as the DiT patch lattice
- after flattening, token index `n` is aligned with image patch index `n`

## 5. Style Encoder

Implemented by `StyleEncoder` and `SourcePartRefDiT.encode_style_global`.

For each sample, every style reference glyph is encoded independently to a `4x4x786` feature map. The `4x4` spatial grid from all valid references is flattened and concatenated into `R*16` local tokens, pooled once by a single-query attention pool without positional embedding and with `style_ref_mask` expanded to token-level masking, then projected by `style_proj: 786 -> 512`.

Shape trace for `B=1`, `R=6`:

| stage | shape |
| --- | --- |
| `style_input` | `(1, 6, 1, 128, 128)` |
| `flatten_refs` | `(6, 1, 128, 128)` |
| `downsample_0` | `(6, 64, 64, 64)` |
| `resblock_0` | `(6, 64, 64, 64)` |
| `downsample_1` | `(6, 128, 32, 32)` |
| `resblock_1` | `(6, 128, 32, 32)` |
| `downsample_2` | `(6, 256, 16, 16)` |
| `resblock_2` | `(6, 256, 16, 16)` |
| `downsample_3` | `(6, 384, 8, 8)` |
| `resblock_3` | `(6, 384, 8, 8)` |
| `downsample_4` | `(6, 786, 4, 4)` |
| `resblock_4` | `(6, 786, 4, 4)` |
| `flatten(2).transpose(1,2)` | `(6, 16, 786)` |
| `view(B, R*16, D)` | `(1, 96, 786)` |
| `expand style_ref_mask to token mask` | `(1, 96)` |
| `style_pool.query` | `(1, 1, 786)` |
| `style_pool.attn` | `(1, 1, 786)` |
| `squeeze(1)` | `(1, 786)` |
| `style_proj` | `(1, 512)` |

Exact aggregation:

```text
style_features_r = style_encoder(style_refs_r)              # [B*R, 786, 4, 4]
style_tokens_r = flatten_hw(style_features_r)               # [B*R, 16, 786]
style_tokens = reshape(style_tokens_r)                      # [B, R*16, 786]
token_mask = expand_ref_mask(style_ref_mask, 16)            # [B, R*16]

pooled_style = style_pool(style_tokens, token_mask)         # [B, 786]
style_global = style_proj(pooled_style)                     # [B, 512]
```

Runtime constraint:

- `style_ref_mask` must keep at least one valid reference per sample
- `style_pool` uses one learned query and no positional embedding, so all valid `R*16` spatial tokens are pooled as an unordered set

## 6. Diffusion Transformer Backbone

Implemented by `DiffusionTransformerBackbone` and `GlyphDiTBlock` in `models/diffusion_transformer_backbone.py`.

### 6.1 Patch embedding and timestep embedding

```text
x = patch_embed(x_t_image).flatten(2).transpose(1,2)         # [B, 64, 512]
x = x + pos_embed                                            # fixed 2D sin-cos, [1, 64, 512]
time_cond = LayerNorm(time_mlp(timestep_embedding(t, 512)))  # [B, 512]
```

### 6.2 Layer plan

Current default block usage:

| block index | 1-based layer | content injection | style injection |
| --- | --- | --- | --- |
| `0` | `1` | yes | no |
| `1` | `2` | yes | no |
| `2` | `3` | yes | no |
| `3` | `4` | yes | no |
| `4` | `5` | yes | no |
| `5` | `6` | yes | no |
| `6` | `7` | no | yes |
| `7` | `8` | no | yes |
| `8` | `9` | no | yes |
| `9` | `10` | no | yes |
| `10` | `11` | no | yes |
| `11` | `12` | no | yes |

### 6.3 One `GlyphDiTBlock` forward order

For one block `l`, input `x_l: [B, 64, 512]`, content tokens `c: [B, 64, 512]`, style global `s: [B, 512]`, timestep condition `t: [B, 512]`.

#### 6.3.1 Content injection

Enabled only if layer `l` is in `content_injection_layers`.

```text
delta_c = zero_linear_l(content_control_ln_l(c))     # [B, 64, 512]
x_l = x_l + delta_c
```

Important details:

- `zero_linear_l` is zero-initialized, so this residual starts from zero contribution
- `c.shape` must exactly match `x_l.shape`

#### 6.3.2 Style injection

Enabled only if layer `l` is in `style_injection_layers`.

```text
delta_s = zero_linear_s_l(style_control_ln_l(s)).unsqueeze(1)  # [B, 1, 512]
x_l = x_l + delta_s
```

Important details:

- style is a single global vector per sample
- this residual is broadcast to all patch tokens
- `zero_linear_s_l` is zero-initialized, so style injection starts from zero contribution
- style does not modify AdaLN parameters

#### 6.3.3 Standard time-conditioned DiT block

After content/style injection, the block runs one self-attention sublayer and one MLP sublayer, both modulated only by `time_cond`.

```text
shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn =
    time_modulation_l(time_mod_input_ln_l(time_cond))        # each [B, 512]

h = modulate(self_ln_l(x_l), shift_sa, scale_sa)             # [B, 64, 512]
x_l = x_l + gate_sa[:, None, :] * self_attn_l(h, h, h)       # [B, 64, 512]

h = modulate(mlp_ln_l(x_l), shift_ffn, scale_ffn)            # [B, 64, 512]
x_{l+1} = x_l + gate_ffn[:, None, :] * mlp_l(h)             # [B, 64, 512]
```

Where:

```text
modulate(x, shift, scale) = x * (1 + scale[:, None, :]) + shift[:, None, :]
```

`time_modulation_l` is zero-initialized on its final linear layer.

### 6.4 Backbone output

After all 12 blocks:

```text
patch_tokens = final_norm(x_12)      # [B, 64, 512]
```

## 7. Patch Detailer Head

Implemented by `PatchDetailerHead` in `models/source_part_ref_dit.py`.

Purpose:

- refine each noisy `16x16` patch independently using the corresponding final transformer token
- inject the token only at the `1x1` bottleneck of a shallow local U-Net

Shape trace for `B=1`, `N=64`:

| stage | shape |
| --- | --- |
| `noisy_patches` | `(1, 64, 1, 16, 16)` |
| `patch_tokens` | `(1, 64, 512)` |
| `flat_patches` | `(64, 1, 16, 16)` |
| `flat_tokens` | `(64, 512)` |
| `enc_block_0` | `(64, 32, 16, 16)` |
| `downsample_0` | `(64, 32, 8, 8)` |
| `enc_block_1` | `(64, 64, 8, 8)` |
| `downsample_1` | `(64, 64, 4, 4)` |
| `enc_block_2` | `(64, 128, 4, 4)` |
| `downsample_2` | `(64, 128, 2, 2)` |
| `enc_block_3` | `(64, 256, 2, 2)` |
| `downsample_3` | `(64, 256, 1, 1)` |
| `context_proj(flat_tokens).view(...,1,1)` | `(64, 384, 1, 1)` |
| `concat_context` | `(64, 640, 1, 1)` |
| `bottleneck` | `(64, 384, 1, 1)` |
| `upsample_0 + concat_skip_0 + dec_block_0` | `(64, 256, 2, 2)` |
| `upsample_1 + concat_skip_1 + dec_block_1` | `(64, 128, 4, 4)` |
| `upsample_2 + concat_skip_2 + dec_block_2` | `(64, 64, 8, 8)` |
| `upsample_3 + concat_skip_3 + dec_block_3` | `(64, 32, 16, 16)` |
| `out_proj` | `(64, 1, 16, 16)` |
| `pred_patches` | `(1, 64, 1, 16, 16)` |

The final `out_proj` convolution is zero-initialized.

## 8. Flow Training Objective

Implemented by `FlowTrainer._compute_losses` in `models/model.py`.

### 8.1 Flow matching path

```text
x1 = target
x0 ~ Normal(0, I)
t ~ Uniform(t_eps, 1 - t_eps), where t_eps = 0.02
x_t = (1 - t) * x0 + t * x1
target_flow = x1 - x0

content_tokens = content_proj(encode_content_tokens(content))
style_global = encode_style_global(style_img, style_ref_mask)
pred_flow = predict_flow(x_t, t, content_tokens=content_tokens, style_global=style_global)
pred_target = x_t + (1 - t) * pred_flow
```

### 8.2 Loss terms

Total loss:

```text
loss =
    flow_term
  + perceptual_term
  + style_term
  + style_batch_term
  + pixel_term
```

Base terms:

```text
loss_flow = mean_i(mean_pixels((pred_flow_i - target_flow_i)^2))
flow_term = lambda_flow * loss_flow

loss_pixel_i = mean_pixels(|pred_target_i - target_i|)
loss_pixel = mean_i(loss_pixel_i)
pixel_term = mean_i(pixel_weight_i * loss_pixel_i)
```

Frozen perceptor guidance terms, enabled only when `use_cnn_perceptor=True` and a perceptor checkpoint is loaded:

```text
loss_perceptual_i = sum_stage mean(|P_stage(pred_target_i) - P_stage(target_i)|)
loss_style_embed_i = 1 - cosine(P_style(pred_target_i), P_style(target_i))

perceptual_term = mean_i(perceptual_weight_i * loss_perceptual_i)
style_term = mean_i(style_weight_i * loss_style_embed_i)
```

Style batch contrastive term:

```text
loss_style_batch = supervised_contrastive_loss(style_global, font_id)
style_batch_term = style_batch_supcon_lambda * loss_style_batch
```

Logged style-global similarity diagnostics:

```text
style_pos_cos, style_neg_cos, style_cos_margin, style_pos_pairs, style_neg_pairs
    = style_similarity_stats(style_global, font_id)
```

Logged conditioning-injection diagnostics:

```text
block_{l}_content_ratio
block_{l}_style_ratio
```

Where:

```text
block_{l}_content_ratio = rms(zero_linear_l(LN(content_tokens))) / rms(block_input_tokens_l)
block_{l}_style_ratio   = rms(zero_linear_l(LN(style_global))) / rms(block_input_tokens_l)
```

These diagnostics are computed immediately before the DiT block and therefore do not require disabling Flash Attention or changing the attention backend.

Runtime detail:

- `style_similarity_stats` no longer builds a cosine matrix; it linearly scans the batch and evaluates at most `64` positive pairs and `64` negative pairs, so `style_pos_pairs` and `style_neg_pairs` are sampled pair counts rather than full-batch pair counts

Time-dependent auxiliary weights:

```text
w(t; lambda, k, m) = lambda * normalized_sigmoid(k * (t - m))
perceptual_weight_i = w(t_i; perceptual_loss_lambda, aux_loss_t_logistic_steepness, perceptual_loss_t_midpoint)
style_weight_i      = w(t_i; style_loss_lambda,      aux_loss_t_logistic_steepness, style_loss_t_midpoint)
pixel_weight_i      = w(t_i; pixel_loss_lambda,      aux_loss_t_logistic_steepness, pixel_loss_t_midpoint)
```

Optimizer learning-rate schedule:

```text
lr_scale(step) = hold-then-linear schedule with warmup, optional decay start, and min scale
optimizer_lr(step) = base_lr * lr_scale(step)
```

### 8.3 Sampling path

Implemented by `FlowTrainer.flow_sample`.

```text
sample_0 ~ Normal(0, I)
dt = 1 / num_inference_steps
content_tokens = content_proj(encode_content_tokens(content))
style_global = encode_style_global(style_img, style_ref_mask)

for k in 0..num_inference_steps-1:
    t_k = k / num_inference_steps
    pred_flow_k = predict_flow(sample_k, t_k, content_tokens=content_tokens, style_global=style_global)
    sample_{k+1} = sample_k + dt * pred_flow_k

sample = clamp(sample_K, -1, 1)
```

## 9. Font Perceptor Architecture

Implemented by `FontPerceptor` in `models/font_perceptor.py`.

Default configuration:

| item | value |
| --- | --- |
| `in_channels` | `1` |
| `base_channels` | `32` |
| `proj_dim` | `128` |
| `num_chars` | `1000` |
| `dropout` | `0.0` |
| `feature_stage_names` | `["stage1", "stage2", "stage3", "stage4"]` |

Parameter counts from `inspect_font_perceptor.py`:

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

Forward shape trace for `B=2`:

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

Outputs:

- `feature_maps`: selected intermediate CNN stages for perceptual loss
- `global_feat`: pooled feature before heads
- `style_embed`: normalized style embedding
- `char_logits`: character classifier logits

## 10. Font Perceptor Pretraining Objective

Implemented by `FontPerceptorTrainer._compute_losses`.

```text
outputs = FontPerceptor(target)
loss_char_ce = CrossEntropy(outputs["char_logits"], char_id)
loss_style_supcon = supervised_contrastive_loss(outputs["style_embed"], font_id, temperature=style_temperature)
loss = loss_char_ce + style_supcon_lambda * loss_style_supcon
```

Validation additionally reports:

```text
char_acc
style_pos_cos
style_neg_cos
style_cos_margin = style_pos_cos - style_neg_cos
```

`train_font_perceptor.py` writes `qualification_report.json` and stamps the same qualification payload into `best.pt` and `last.pt`.

## 11. Current CLI Surface

### 11.1 Main flow training

`train.py` now accepts the current conditioning layer controls:

```bash
--content-injection-layers 1,2,3,4,5,6
--style-injection-layers 7,8,9,10,11,12
```

Old runtime parameters removed from the runnable model path:

- `--content-cross-attn-heads`
- `--content-cross-attn-layers`
- `--style-cross-attn-layers`
- `--style-modulation-layers`

### 11.2 Helper scripts

`scripts/run_diffusion_colab.sh` and `scripts/profile_memory.py` use the same new layer names.

`scripts/inspect_flow_model.py` traces:

- content path
- style global-vector path
- backbone block-by-block conditioning plan
- patch detailer path

`scripts/inspect_font_perceptor.py` traces:

- CNN stage outputs
- pooled global feature
- style embedding
- character logits

## 12. Code Map

| component | file |
| --- | --- |
| main flow entry | `train.py` |
| perceptor pretrain entry | `train_font_perceptor.py` |
| dataset and samplers | `dataset.py` |
| main generator wrapper, content/style encoders, patch detailer | `models/source_part_ref_dit.py` |
| DiT backbone and conditioning blocks | `models/diffusion_transformer_backbone.py` |
| SDPA attention wrapper | `models/sdpa_attention.py` |
| flow trainer and perceptor trainer | `models/model.py` |
| font perceptor model and frozen guidance wrapper | `models/font_perceptor.py` |
| flow model shape inspector | `scripts/inspect_flow_model.py` |
| perceptor shape inspector | `scripts/inspect_font_perceptor.py` |
