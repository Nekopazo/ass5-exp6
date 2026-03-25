# Archived

This document describes the removed dual-branch `global_style + style_tokens` architecture and is kept only for history.
Current code follows [docs/i2i_unified_k_style_memory_compact.md](/scratch/yangximing/code/ass5-exp6/DiffuFont/docs/i2i_unified_k_style_memory_compact.md): single 32x32 style encoder, unified `K` style memory, no explicit global/local split.

# DiffuFont Current Model Architecture

This document describes the current code path in `/scratch/yangximing/code/ass5-exp6/DiffuFont` after the latest style-branch refactor.

It follows the current main flow setup:

- `latent_channels = 12`
- `latent_size = 16`
- `vae_bottleneck_channels = 192`
- `dit_depth = 12`
- `content_cross_attn_indices = [0, 1, 2, 3, 4, 5, 8, 10]`
- `style_tokens_per_ref = 8`
- `style_ref_count = 8`

If `train.py` defaults and shell script defaults differ, this document follows the active main training script snapshot rather than the bare constructor defaults.

## 1. Top-Level Structure

`SourcePartRefDiT` contains four major parts:

1. `GlyphVAE`
2. `content_encoder`
3. `global_style_encoder` + `token_style_encoder` + style assembly branch
4. `DiffusionTransformerBackbone`

High-level data flow in flow training:

```text
content image (B,1,128,128)
  -> content_encoder
  -> content_tokens (B,256,512)

style refs (B,8,1,128,128)
  -> global_style_encoder
  -> token_style_encoder
  -> style_global (B,512)
  -> style_tokens (B,64,512)

target image (B,1,128,128)
  -> VAE encode
  -> latent z1 (B,12,16,16)

noise z0 + interpolation zt
  -> DiT backbone with:
     - content cross-attn on selected blocks
     - style cross-attn on every block
     - global style added into timestep condition
  -> pred_flow (B,12,16,16)
```

## 2. Parameter Snapshot

These counts were measured by instantiating the current model with the config above.

| Module | Params |
|---|---:|
| Full model | 152,441,381 |
| VAE | 3,138,073 |
| Content encoder | 12,643,840 |
| Global style encoder | 12,644,864 |
| Token style encoder | 12,643,840 |
| Local style adapter | 1,319,936 |
| Global style pool | 1,053,184 |
| Per-ref style pool | 1,056,768 |
| DiT backbone | 107,940,876 |

Current main-path tensor shapes from smoke test:

- `content_tokens = (B, 256, 512)`
- `style_tokens = (B, 64, 512)`
- `style_global = (B, 512)`
- `style_token_mask = (B, 64)`
- `pred_flow = (B, 12, 16, 16)`

## 3. VAE

### 3.1 Encoder

Input:

- glyph image: `(B, 1, 128, 128)`

Structure:

```text
Conv 3x3 s2:    1   -> 32   => (B,32,64,64)
ResBlock:       32  -> 32
Conv 3x3 s2:    32  -> 64   => (B,64,32,32)
ResBlock:       64  -> 64
Conv 3x3 s2:    64  -> 192  => (B,192,16,16)
ResBlock:       192 -> 192
Extra ResBlock: 192 -> 192   (encoder_16x16_blocks = 2)
Stats head 1x1: 192 -> 24    => split into mu/logvar
```

Latent stats:

- `mu = (B, 12, 16, 16)`
- `logvar = (B, 12, 16, 16)`
- sampled or mean latent `z = (B, 12, 16, 16)`

### 3.2 Decoder

Input:

- latent: `(B, 12, 16, 16)`

Structure:

```text
Conv 3x3:       12  -> 192   => (B,192,16,16)
ResBlock:       192 -> 192
Extra ResBlock: 192 -> 192   (decoder_16x16_blocks = 2)

Upsample x2 + Conv: 192 -> 64 => (B,64,32,32)
ResBlock:           64  -> 64

Upsample x2 + Conv: 64  -> 32 => (B,32,64,64)
ResBlock:           32  -> 32

Upsample x2 + Conv: 32  -> 16 => (B,16,128,128)
Tail ResBlock:      16  -> 16  (decoder_tail_blocks = 1)

GroupNorm + SiLU + Conv 3x3: 16 -> 1
Tanh
```

Output:

- reconstruction: `(B, 1, 128, 128)`

### 3.3 VAE Loss

Current VAE training uses:

- `loss_rec`: pixel L1 reconstruction
- `loss_perc`: custom glyph perceptual loss
- `loss_kl`: KL regularization

Current glyph perceptual loss is multi-scale and already includes gradient matching at scales `1 / 2 / 4`, so there is no separate extra edge-loss branch now.

## 4. Content Branch

`content_encoder` is a patch-token encoder without CLS token.

Input:

- content image: `(B, 1, 128, 128)`

Patch setup:

- patch size `8`
- grid size `16 x 16`
- token count `256`
- hidden dim `512`

Flow:

```text
Conv patch_embed 8x8 s8:
  (B,1,128,128) -> (B,512,16,16)
flatten -> (B,256,512)
+ 2D sin-cos positional embedding
+ 4 x EncoderBlock
+ final LayerNorm
```

Output:

- `content_tokens = (B, 256, 512)`

These tokens preserve explicit patch order throughout the content path.

## 5. Style Branch

## 5.1 Design Intent

The style branch is now explicitly split into two roles:

1. `style_global`
   - global font identity / overall style
   - supervised by contrastive learning during style pretraining
   - frozen during flow training by default

2. `style_tokens`
   - local pen shape, curvature, stroke ending, decorations
   - no NCE supervision
   - trained in flow together with the main backbone

The contrastive head is no longer part of the runtime model. It now lives inside `StylePretrainTrainer` only.

## 5.2 Separate Style Encoders

The style branch now uses two different encoders.

### Global Style Encoder

`global_style_encoder` includes a CLS-style global token.

Per reference glyph input:

- `(B*8, 1, 128, 128)`

Outputs:

- local patch tokens: `(B*8, 256, 512)` (unused by the global path)
- global token: `(B*8, 512)`

After reshape:

- global tokens: `(B, 8, 512)`

This encoder is the one learned by style pretraining and frozen in flow by default.

### Token Style Encoder

`token_style_encoder` has no CLS token and only produces local patch tokens.

Per reference glyph input:

- `(B*8, 1, 128, 128)`

Outputs:

- local patch tokens: `(B*8, 256, 512)`

After reshape:

- local tokens: `(B, 8, 256, 512)`

This encoder is not touched by style pretraining. It is trained directly by the flow objective.

## 5.3 Global Style Path

Global path uses only the `global_style_encoder` outputs:

```text
(B,8,512)
-> AttentionTokenPool(output_tokens=1)
-> (B,1,512)
-> squeeze
-> style_global = (B,512)
```

This is the only path used by style pretraining.

Style pretraining objective:

```text
style_global
-> trainer-owned contrastive_head
-> 128-d embedding
-> InfoNCE(anchor, positive)
```

Checkpoint payload for style pretraining now contains:

- `global_style_state`
- `contrastive_head_state`

Flow training only loads `global_style_state`.

## 5.4 Local Style Token Path

Local path uses only `token_style_encoder` local patch tokens.

### Step A: Local token grid

Per reference:

- input local tokens: `(B, 8, 256, 512)`

Each reference still corresponds to a `16 x 16` patch grid.

### Step B: LocalStyleAdapter

Before pooling, each reference passes through `LocalStyleAdapter`.

For each reference independently:

```text
(256,512)
-> reshape to (512,16,16)
-> depthwise 3x3 conv
-> pointwise 1x1 conv
-> residual
-> LayerNorm + MLP residual
-> back to (256,512)
```

Purpose:

- inject local spatial refinement before pooling
- bias tokens toward local stroke geometry instead of pure global summarization

### Step C: Per-reference pooling

Each reference is pooled independently:

```text
(B*8,256,512)
-> AttentionTokenPool(output_tokens=8)
-> (B*8,8,512)
```

Reshape:

```text
(B*8,8,512)
-> (B,8,8,512)
-> flatten ref dimension
-> (B,64,512)
```

Then:

- optional projection `style_proj` if hidden dims differ
- mask expansion from `(B,8)` to `(B,64)`

Final outputs:

- `style_tokens = (B, 64, 512)`
- `style_token_mask = (B, 64)`

## 5.5 Does This Pooling Preserve Image Position?

Short answer:

- it preserves position implicitly
- it does not preserve an explicit spatial grid after pooling

More precisely:

1. Before pooling, local patch tokens carry explicit 2D sin-cos position information from `token_style_encoder`.
2. `LocalStyleAdapter` also operates on the real `16 x 16` token grid, so it uses explicit local neighborhood structure.
3. After `256 -> 8` pooling, explicit `(x, y)` alignment is gone.
4. What remains is:
   - per-reference grouping
   - position-aware token content
   - learned local prototypes

So `style_tokens` are not a spatial feature map. They are position-aware local style prototypes grouped by reference.

This is intentionally different from `style_global`:

- `style_global` is a single vector shared across every latent token
- `style_tokens` are a cross-attention memory bank that different latent positions can query differently

## 6. DiT Backbone

Current backbone config:

- latent channels: `12`
- latent size: `16`
- latent tokens: `256`
- hidden dim: `512`
- depth: `12`
- heads: `8`
- MLP ratio: `4.0`

Current block layout:

- content+style blocks: `[0, 1, 2, 3, 4, 5, 8, 10]`
- style-only blocks: `[6, 7, 9, 11]`

Style cross-attention runs on every block.

### 6.1 Latent Input

Input latent:

- `x_t = (B, 12, 16, 16)`

Flow:

```text
flatten -> (B,256,12)
Linear 12 -> 512
+ 2D latent positional embedding
=> latent tokens (B,256,512)
```

### 6.2 Conditioning

Timestep:

- `t -> timestep_embedding(512) -> time_mlp -> (B,512)`

Global style:

- `style_global -> global_style_cond_proj -> (B,512)`

Combined global condition:

- `global_cond = time_embed + global_style_cond_proj(style_global)`

This `global_cond` drives:

- self-attention modulation
- content cross-attention modulation
- MLP modulation

It does not directly control style cross-attention strength.

Style cross-attention is now the cleanest version:

- query comes directly from normalized latent states
- key/value come directly from `style_tokens`
- there is no extra token-derived style modulation branch

### 6.3 Per-Block Logic

Each block always contains:

- self-attention
- MLP
- global modulation from `global_cond`

Some blocks additionally contain:

- content cross-attention if index is in `content_cross_attn_indices`
- style cross-attention if style is enabled for that layer

Current role split inside each block:

- `style_global` sets the global block operating context through `global_cond`
- `style_tokens` only provide the cross-attention memory itself

In the current config:

- blocks `0,1,2,3,4,5,8,10` read both `content_tokens` and `style_tokens`
- blocks `6,7,9,11` read only `style_tokens`

The style-only blocks do not instantiate content attention parameters.

### 6.4 Output

After 12 blocks:

```text
LayerNorm
Linear 512 -> 12
reshape
-> pred_flow = (B,12,16,16)
```

## 7. Stage-Specific Training Behavior

## 7.1 VAE Stage

Active modules:

- `vae`

Inactive for loss:

- content encoder
- global style encoder
- token style encoder
- style pools
- DiT backbone

## 7.2 Style Pretraining Stage

Active modules:

- `global_style_encoder`
- `global_style_pool`
- `style_global_proj`
- trainer-owned `contrastive_head`

Skipped:

- `token_style_encoder`
- `local_style_adapter`
- `per_ref_style_pool`
- `style_tokens`
- DiT backbone

Important consequence:

- style pretraining no longer teaches `style_tokens`
- it only learns a frozen global style extractor

## 7.3 Flow Stage

Typical current setup:

- VAE frozen
- global style extractor frozen
- local token style branch trainable
- DiT backbone trainable

Trainable token-style branch:

- `token_style_encoder`
- `local_style_adapter`
- `per_ref_style_pool`
- `style_proj`

Frozen global branch:

- `global_style_encoder`
- `global_style_pool`
- `style_global_proj`

This is exactly the intended split:

- `style_global` stays stable and acts as the frozen global font identity signal
- `style_tokens` keep adapting to local visual detail needed by the generation task

## 8. Why This Style Split Is Different From The Old Design

Old behavior pushed pooled style tokens toward a contrastive global summary.

Current behavior separates responsibilities:

- global contrastive supervision only touches `style_global`
- local pooled tokens are free to specialize for generation
- per-reference grouping is preserved until after each ref has already produced its own `8` tokens

This makes the token branch more suitable for:

- stroke endings
- curvature
- local pen thickness variation
- decorative details

without forcing those tokens to collapse into a single font-level identity embedding.
