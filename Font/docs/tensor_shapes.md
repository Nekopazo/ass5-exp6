# Model Data Flow

This document only describes the internal tensor flow of the current `SourcePartRefDiT` model.

## Notation

```text
B    number of generated target glyphs in the batch
Uc   number of unique content glyphs in the batch
Us   number of unique style fonts in the batch
R    number of style reference glyphs per style font
T    token count on the 16x16 grid = 256
De   encoder dim = 256
Dt   DiT hidden dim = 256
```

For the default full training batch:

```text
B  = 64 fonts * 6 chars = 384
Uc = 6
Us = 64
R  = 4..8, fixed inside one batch
```

## 1. Inputs Enter The Model

The collator sends compacted condition tensors plus index tensors:

```text
target:          (B, 3, 128, 128)
content:         (Uc, 3, 128, 128)
content_index:   (B,)
style_img:       (Us, R, 3, 128, 128)
style_index:     (B,)
```

`content_index` expands unique content glyphs to the `B` target samples.

`style_index` expands unique style banks to the `B` target samples.

There is no style mask. All `R` style references in `style_img` are valid.

## 2. Diffusion Noising

Training creates the noisy image before the DiT forward:

```text
x1 = target:             (B, 3, 128, 128)
x0 = gaussian noise:     (B, 3, 128, 128)
t:                       (B,)
t_view:                  (B, 1, 1, 1)
x_t = t*x1 + (1-t)*x0:   (B, 3, 128, 128)
target_velocity = x1-x0: (B, 3, 128, 128)
```

`x_t` is the image input to the DiT backbone.

## 3. Content Encoder

The content encoder processes only unique content glyphs:

```text
content:                 (Uc, 3, 128, 128)
ResDown 3 -> 64:         (Uc, 64, 64, 64)
ResDown 64 -> 128:       (Uc, 128, 32, 32)
ResDown 128 -> 256:      (Uc, 256, 16, 16)
tail GN/SiLU/1x1:        (Uc, 256, 16, 16)
flatten tokens:          (Uc, 256, 256)
```

Then the unique content tokens are expanded:

```text
unique_content_tokens:   (Uc, T, De)
content_tokens:          (B, T, De)
```

With defaults:

```text
unique_content_tokens:   (6, 256, 256)
content_tokens:          (384, 256, 256)
```

## 4. Style Encoder

The style encoder processes unique style banks:

```text
style_img:               (Us, R, 3, 128, 128)
flatten refs:            (Us*R, 3, 128, 128)
ResDown 3 -> 64:         (Us*R, 64, 64, 64)
ResDown 64 -> 128:       (Us*R, 128, 32, 32)
ResDown 128 -> 256:      (Us*R, 256, 16, 16)
tail GN/SiLU/1x1:        (Us*R, 256, 16, 16)
flatten tokens:          (Us*R, 256, 256)
reshape bank:            (Us, R, 256, 256)
```

The content and style encoders have the same architecture but separate weights.

## 5. Content-Style Cross Attention

Content tokens provide queries. Style tokens provide keys and values.

Content query path:

```text
unique_content_tokens:   (Uc, T, De)
unique_content_query:    (Uc, 4, T, 64)
content_query:           (B, 4, T, 64)
```

Style key/value path:

```text
style_token_bank:        (Us, R, T, De)
concat style tokens:     (Us, R*T, De)
style_key:               (Us, 1, 4, R*T, 64)
style_value:             (Us, 1, 4, R*T, 64)
expanded style_key:      (B, 1, 4, R*T, 64)
expanded style_value:    (B, 1, 4, R*T, 64)
```

Attention runs after flattening the single style-bank dimension:

```text
query:                   (B, 4, T, 64)
key:                     (B, 4, R*T, 64)
value:                   (B, 4, R*T, 64)
style_context:           (B, T, De)
```

The final conditioning tokens concatenate content and style context:

```text
content_tokens:          (B, T, De)
style_context:           (B, T, De)
conditioning_tokens:     (B, T, 512)
```

With defaults:

```text
conditioning_tokens:     (384, 256, 512)
```

## 6. DiT Patch Path

The noisy image is patch embedded:

```text
x_t:                     (B, 3, 128, 128)
patch embed conv:        (B, Dt, 16, 16)
flatten patches:         (B, T, Dt)
add pos embed:           (B, T, Dt)
```

With defaults:

```text
patch_tokens:            (384, 256, 256)
```

## 7. Conditioning Inside Each DiT Block

Each DiT block receives:

```text
patch_tokens:            (B, T, Dt)
timestep embedding:      (B, Dt)
conditioning_tokens:     (B, T, 512)
```

The conditioning tokens are split:

```text
content condition:       (B, T, 256)
style condition:         (B, T, 256)
```

They are projected to DiT hidden dim and added with time:

```text
content_hidden:          (B, T, Dt)
style_hidden:            (B, T, Dt)
time_hidden:             (B, 1, Dt)
joint_hidden:            (B, T, Dt)
```

`joint_hidden` produces modulation for the self-attention and MLP:

```text
modulation:              (B, T, 6*Dt)
shift/scale/gate x2:     six tensors of (B, T, Dt)
```

The block output keeps the same shape:

```text
patch_tokens:            (B, T, Dt)
```

After all DiT blocks:

```text
patch_tokens:            (B, T, Dt)
```

## 8. Output Head

The final head again uses time and conditioning:

```text
patch_tokens:            (B, T, Dt)
conditioning_tokens:     (B, T, 512)
output_condition_hidden: (B, T, Dt)
output_time_hidden:      (B, 1, Dt)
joint_hidden:            (B, T, Dt)
output shift/scale:      two tensors of (B, T, Dt)
```

Patch pixels are predicted:

```text
output_proj:             (B, T, 3*8*8)
patch pixels:            (B, 16, 16, 3, 8, 8)
prediction image:        (B, 3, 128, 128)
```

## 9. Loss

The current model predicts `x`.

```text
prediction = pred_x:     (B, 3, 128, 128)
pred_velocity:           (B, 3, 128, 128)
target_velocity:         (B, 3, 128, 128)
loss = MSE(pred_velocity, target_velocity)
```

`pred_x_l1` is only logged for monitoring.

## 10. Sampling

During sampling, conditioning is encoded once:

```text
content/style inputs -> conditioning_tokens: (B, T, 512)
condition cache per DiT block:               (B, T, Dt)
output condition hidden:                     (B, T, Dt)
```

The ODE loop updates an image tensor:

```text
sample image:            (B, 3, 128, 128)
predicted velocity:      (B, 3, 128, 128)
updated sample image:    (B, 3, 128, 128)
```

The saved training sample grid uses:

```text
16 content images
16 target images
16 generated images
```

