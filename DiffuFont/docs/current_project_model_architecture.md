# Current Project Model Architecture

Generated from the current repository state in `/scratch/yangximing/code/ass5-exp6/DiffuFont`.

This document describes only the current model. Historical branches and removed alternatives are intentionally omitted.

## 1. Main Path

Current model:

- model: `SourcePartRefDiT`
- input/output channels: `1`
- content encoder output: `8x8x512`
- style encoder output: `8x8x512`
- content/style encoders: shared CNN pyramid template
- style fusion: one-shot `content <- style` cross-attention outside the backbone
- DiT feed-forward: `swiglu`
- DiT normalization/modulation: `rms`
- DiT patch size: `16`
- DiT hidden size: `512`
- DiT heads: `8`
- content-style cross-attention heads: `4`
- optimizer: `AdamW`
- default `weight_decay`: `0.01`

Effective path:

1. Encode `content_img` into `8x8x512` content tokens.
2. Encode all style references into `8x8x512` style tokens.
3. Concatenate all style-reference tokens along the token axis.
4. Run one cross-attention with content tokens as query and style tokens as key/value.
5. Add the resulting style context back to content tokens.
6. Feed fused content tokens into the DiT backbone.
7. Use timestep-conditioned RMS modulation in every DiT block.
8. Refine patch outputs with `PatchDetailerHead`.
9. Unpatchify patch outputs into final image-space flow.

There is no global style vector path in the current model.

## 2. Main Configuration

| item | value |
| --- | --- |
| `in_channels` | `1` |
| `image_size` | `128` |
| `patch_size` | `16` |
| `patch_grid_size` | `8` |
| `num_patches` | `64` |
| `encoder_hidden_dim` | `512` |
| `dit_hidden_dim` | `512` |
| `dit_depth` | runtime-configured |
| `dit_heads` | `8` |
| `dit_mlp_ratio` | `4.0` |
| `ffn_activation` | `swiglu` |
| `norm_variant` | `rms` |
| `content_style_fusion_heads` | `4` |
| `content_injection_layers` | all DiT layers |
| `detailer_base_channels` | runtime-configured |
| `detailer_max_channels` | runtime-configured |
| `detailer_bottleneck_channels` | `512` |
| `weight_decay` | `0.01` |

## 3. Top-Level Forward Graph

Notation:

- `B`: batch size
- `R`: number of style references
- `H = W = 128`
- `P = 16`
- `N = (H / P)^2 = 64`
- `D = 512`

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
content_tokens = encode_content_tokens(content_img)                 # [B, 64, 512]
content_tokens = content_proj(content_tokens)                       # [B, 64, 512]

style_token_bank, token_valid_mask =
    encode_style_token_bank(style_img, style_ref_mask)              # [B, R*64, 512], [B, R*64]

fused_content_tokens = fuse_content_style_tokens(
    content_tokens,
    style_token_bank,
    token_valid_mask=token_valid_mask,
)                                                                   # [B, 64, 512]

patch_tokens = backbone(
    x_t_image,
    timesteps,
    content_tokens=fused_content_tokens,
)                                                                   # [B, 64, 512]

noisy_patches = patchify(x_t_image)                                 # [B, 64, 1, 16, 16]
pred_patches = detailer(patch_tokens, noisy_patches)                # [B, 64, 1, 16, 16]
pred_flow = unpatchify(pred_patches)                                # [B, 1, 128, 128]
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
| downsample stage 3 | `[B, 512, 8, 8]` |
| residual block 3 | `[B, 512, 8, 8]` |
| flatten to tokens | `[B, 64, 512]` |

Semantics:

- content tokens are spatially aligned with the `8x8` DiT patch lattice
- token index matches patch index

## 5. Style Path

Implemented by `StyleEncoder` and `ContentStyleCrossAttention` in `models/source_part_ref_dit.py`.

Each style reference is encoded independently, then regrouped per sample.

The style encoder now uses the same CNN pyramid template as the content encoder:

- stem conv
- stem block
- three stride-2 pyramid stages
- final `8x8x512` feature map

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
| downsample stage 3 | `[B*R, 512, 8, 8]` |
| residual block 3 | `[B*R, 512, 8, 8]` |
| flatten per ref | `[B*R, 64, 512]` |
| regroup refs | `[B, R, 64, 512]` |
| concat refs | `[B, R*64, 512]` |
| token valid mask | `[B, R*64]` |
| style token projection | `[B, R*64, 512]` |

Outputs:

```text
style_token_bank = [B, R*64, 512]
token_valid_mask = [B, R*64]
```

## 6. External Content-Style Cross Attention

The style fusion happens exactly once outside the backbone.

| tensor | shape |
| --- | --- |
| query = `content_tokens` | `[B, 64, 512]` |
| key = `style_token_bank` | `[B, R*64, 512]` |
| value = `style_token_bank` | `[B, R*64, 512]` |
| `style_context` | `[B, 64, 512]` |
| concat(`content_tokens`, `style_context`) | `[B, 64, 1024]` |
| fusion norm | `[B, 64, 1024]` |
| fused content tokens | `[B, 64, 512]` |

Equation:

```text
style_context = CrossAttention(
    Q=[B, 64, 512],
    K=[B, R*64, 512],
    V=[B, R*64, 512],
)

fused_content = Linear(1024 -> 512)(
    LayerNorm(1024)(
        concat([content_tokens, style_context], dim=-1)
    )
)
```

Properties:

- cross-attention runs only once
- content/style fusion is learned, not residual addition
- multiple style references are concatenated along the token dimension
- no additional style downsampling is applied before cross-attention
- implementation uses `SDPAAttention`
- intended CUDA fast path is flash attention
- current path assumes all style refs are valid in the active batch

## 7. DiT Backbone

Implemented by `DiffusionTransformerBackbone` and `GlyphDiTBlock` in `models/diffusion_transformer_backbone.py`.

### 7.1 Patch and Timestep Embedding

```text
x = patch_embed(x_t_image).flatten(2).transpose(1, 2)      # [B, 64, 512]
x = x + pos_embed                                           # [B, 64, 512]

time_cond = timestep_embedding(timesteps, 512)              # [B, 512]
time_cond = time_mlp(time_cond)                             # [B, 512]
time_cond = time_cond_norm(time_cond)                       # [B, 512]
time_tokens = time_cond.unsqueeze(1).expand(-1, 64, -1)     # [B, 64, 512]
```

### 7.2 Block Conditioning

Backbone-level conditioning is prepared once before the block stack:

```text
conditioning_tokens = time_tokens + fused_content_tokens
```

Each block receives:

- current patch tokens
- conditioning tokens

There is no style-specific conditioning path inside the backbone.
There is also no per-block timestep projection anymore; per-block differences come from each block's own `joint_mod`.

## 8. RMS Modulation

Active normalization bundle:

- `norm_variant="rms"`
- per branch parameters: `scale` and `gate`
- all active norm layers use `elementwise_affine=False`

Per block, a zero-initialized linear head predicts:

```text
self_attn_scale, self_attn_gate, ffn_scale, ffn_gate
```

Rule:

```text
norm_x = RMSNorm(x)
modulated_x = norm_x * (1 + scale)
x = x + gate * branch_out
```

There is no `shift` term in the active path.

Cross-attention token normalization:

- `query_norm = LayerNorm(512, elementwise_affine=False)`
- `token_norm = LayerNorm(512, elementwise_affine=False)`
- normalization is applied once before `q/k/v` projection

## 9. SwiGLU Feed-Forward

Each DiT block uses `FeedForward(..., activation="swiglu")`.

FFN path:

```text
Linear(hidden_dim -> 2 * inner_dim)
split into value and gate
SiLU(gate) * value
Linear(inner_dim -> hidden_dim)
```

With `hidden_dim = 512` and `mlp_ratio = 4.0`:

| stage | shape |
| --- | --- |
| input | `[B, 64, 512]` |
| first linear | `[B, 64, 4096]` |
| split value/gate | `2 x [B, 64, 2048]` |
| SwiGLU output | `[B, 64, 2048]` |
| second linear | `[B, 64, 512]` |

## 10. Full Backbone Shape Trace

Each block preserves token shape.

Inputs to one `GlyphDiTBlock`:

| tensor | shape |
| --- | --- |
| `x_l` | `[B, 64, 512]` |
| `time_cond` | `[B, 512]` |
| `content_tokens` | `[B, 64, 512]` |

Inside one block:

| stage | shape |
| --- | --- |
| timestep token expansion | `[B, 64, 512]` |
| content source after norm | `[B, 64, 512]` |
| joint conditioning source | `[B, 64, 512]` |
| modulation head output | `[B, 64, 2048]` |
| split modulation tensors | `4 x [B, 64, 512]` |

Modulation tensors:

```text
self_attn_scale = [B, 64, 512]
self_attn_gate  = [B, 64, 512]
ffn_scale       = [B, 64, 512]
ffn_gate        = [B, 64, 512]
```

Full backbone:

```text
[B, 64, 512]
 -> block 1
 -> block 2
 -> ...
 -> block D
 -> final RMS norm
 = [B, 64, 512]
```

Backbone output:

```text
patch_tokens = [B, 64, 512]
```

## 11. Patch Detailer

Implemented by `PatchDetailerHead`.

Inputs:

| tensor | shape |
| --- | --- |
| `patch_tokens` | `[B, 64, 512]` |
| `noisy_patches` | `[B, 64, 1, 16, 16]` |

Flattened internal tensors:

| stage | shape |
| --- | --- |
| flat patch images | `[B*64, 1, 16, 16]` |
| flat patch tokens | `[B*64, 512]` |

Current channel plan:

```text
[64, 128, 256, 512]
```

Detailer shape trace:

| stage | shape |
| --- | --- |
| input block (`conv -> SiLU`) | `[B*64, 64, 16, 16]` |
| max pool 1 | `[B*64, 64, 8, 8]` |
| down block 1 (`conv -> SiLU`) | `[B*64, 128, 8, 8]` |
| max pool 2 | `[B*64, 128, 4, 4]` |
| down block 2 (`conv -> SiLU`) | `[B*64, 256, 4, 4]` |
| max pool 3 | `[B*64, 256, 2, 2]` |
| down block 3 (`conv -> SiLU`) | `[B*64, 512, 2, 2]` |
| max pool 4 | `[B*64, 512, 1, 1]` |
| token reshape | `[B*64, 512, 1, 1]` |
| bottleneck concat input | `[B*64, 1024, 1, 1]` |
| bottleneck output | `[B*64, 512, 1, 1]` |
| nearest upsample 1 + skip concat | `[B*64, 1024, 2, 2]` |
| up block 1 (`conv -> SiLU`) | `[B*64, 512, 2, 2]` |
| nearest upsample 2 + skip concat | `[B*64, 768, 4, 4]` |
| up block 2 (`conv -> SiLU`) | `[B*64, 256, 4, 4]` |
| nearest upsample 3 + skip concat | `[B*64, 384, 8, 8]` |
| up block 3 (`conv -> SiLU`) | `[B*64, 128, 8, 8]` |
| nearest upsample 4 + skip concat | `[B*64, 192, 16, 16]` |
| up block 4 (`conv -> SiLU`) | `[B*64, 64, 16, 16]` |
| output projection | `[B*64, 1, 16, 16]` |
| reshape back | `[B, 64, 1, 16, 16]` |

The current detailer has no explicit timestep-conditioning path. Downsampling is performed by a separate `MaxPool2d(2)` after each down block, with no extra activation after the pool. Upsampling is performed by nearest-neighbor resize first, then skip concatenation, then the single-conv block.

Patch-token conditioning is injected directly at the bottleneck by channel concatenation:

```text
token_context = s_i.view(B*64, 512, 1, 1)
x = concat([bottleneck_feature, token_context], dim=1)
```

Final output:

```text
pred_flow = [B, 1, 128, 128]
```

## 12. Training Path

`FlowTrainer` optimizes pixel-space flow prediction.

Optimizer:

- `AdamW`
- two param groups: `decay` and `no_decay`
- `decay` group uses `weight_decay=0.01`
- `no_decay` group uses `weight_decay=0.0`
- `no_decay` covers parameters with `ndim <= 1`, explicit `.bias` parameters, and batch-norm style parameters if present

Because the active norm modules use `elementwise_affine=False`, they do not contribute trainable norm weights or biases.

Active losses:

- flow MSE term
- pixel reconstruction term
- optional perceptual term when a perceptor checkpoint is explicitly enabled

The architecture itself does not depend on the perceptor.

## 13. Gradient Flow For Deduplicated Conditions

Training batches deduplicate content and style encoder inputs before expanding them back to sample-level batches.

Current behavior:

- content/style encoder forward passes run on unique items only
- expanded batch conditioning still drives the full DiT backbone and detailer
- gradients returning to the deduplicated content/style condition banks are averaged by reuse count
- averaged gradients still preserve differences from distinct `x_t / t / target` samples; only the duplicate-count scaling is removed
- backbone and detailer gradients remain full-batch gradients

So:

- encoder-side shared condition tensors use mean gradient over repeated references
- backbone-side modules still optimize against the full effective batch
