# 中文字风格迁移 Diffusion Transformer

## 1. 当前目标

输入一个 content 字形结构图和多张 style reference 图，生成目标字体下的同一个汉字。当前模型是 `SourcePartRefDiT`，训练目标是 x-pred，并从 `x_pred` 推导 `v_pred` 做 loss。

## 2. 当前默认结构

默认配置：

- 图像：`128x128` 灰度图
- patch size：`8`
- token 数：`16x16 = 256`
- content token：`[B, 256, 256]`
- style token bank：`[B, R, 256, 256]`
- 条件 token：`[B, 256, 512]`
- DiT hidden dim：`256`
- DiT block：默认 12 层
- style fusion：固定 `cross_attn_concat`

## 3. 编码流程

```text
content_img -> ContentEncoder -> content_tokens              # [B, 256, 256]
style_img   -> StyleEncoder   -> style_token_bank            # [B, R, 256, 256]
```

content 和 style 都经过卷积金字塔下采样到 `16x16`，再展平成 token。

## 4. 拼接 Cross-Attention

当前模型只保留拼接 cross-attention：先把多张 ref 的 style tokens 拼接，再由 content 查询：

```text
style_tokens_concat = [B, R*256, 256]
Q = Wq(norm(content_tokens))
K,V = Wk/Wv(norm(style_tokens_concat))
style_key_valid_mask = reshape(style_ref_mask, [B, R*256])
style_context = Attention(Q, K, V, key_mask=style_key_valid_mask)  # [B, 256, 256]
conditioning_tokens = concat(content_tokens, style_context)   # [B, 256, 512]
```

含义：每个 content token 会在所有有效 ref 的所有 style token 里查询需要的风格信息。无效或 padding 的 ref token 会被真正从 attention key 里 mask 掉，不再占用 softmax 权重。得到的 `style_context` 仍然是 style value 经过注意力权重重组后的结果，不是原始 content，也不是简单平均。

没有每张 ref 独立查询后的平均分支，也没有 ref-level gate 分支。

## 5. DiT 条件注入

每个 DiT block 都把 `conditioning_tokens` 拆成 content/style 两半：

```text
content_hidden = Linear(RMSNorm(content_tokens))
style_hidden   = Linear(RMSNorm(style_tokens))

time_hidden = Linear(time_cond)
joint_hidden = SiLU(time_hidden + content_hidden + style_hidden)
mods = Linear(joint_hidden)
```

`mods` 产生 self-attention 和 FFN 的 shift、scale、gate：

```text
self_attn_shift, self_attn_scale, self_attn_gate,
ffn_shift,       ffn_scale,       ffn_gate
```

## 6. Output Head

最后输出头也使用同样的三项相加：

```text
content_hidden = Linear(RMSNorm(content_tokens))
style_hidden   = Linear(RMSNorm(style_tokens))
time_hidden    = Linear(time_cond)

joint_hidden = SiLU(time_hidden + content_hidden + style_hidden)
shift, scale = Linear(joint_hidden).chunk(2)

x = RMSNorm(patch_tokens)
x = x * (1 + scale) + shift
patch_pixels = Linear(x)
unpatchify -> pred_x
```

## 7. 推理缓存

推理时，`conditioning_tokens` 不随 denoise step 变化，所以会提前缓存：

- 每个 block 的 content/style condition projection
- output head 的 content/style condition projection

每一步仍然重新算 timestep 相关 modulation。这个缓存只减少重复计算，不改变输出。

sample 可视化不会长期把 sample batch 缓存在 GPU 上；每次到 `sample_every_steps` 都重新构造 batch 并重新推理。style ref 使用固定 seed 选出同一组 ref。

训练阶段可以在 `style_ref_count_min` 到 `style_ref_count_max` 之间随机采样 ref 数；验证和 sample 可视化固定使用 `style_ref_count_max`。

## 8. 训练目标

```text
t = sigmoid(randn * p_std + p_mean)
x0 = noise_scale * randn
xt = t * x1 + (1 - t) * x0

pred_x = model(xt, t, condition)
target_v = (x1 - xt) / clamp_min(1 - t, t_eps)
pred_v   = (pred_x - xt) / clamp_min(1 - t, t_eps)
loss = mse(pred_v, target_v)
```

核心原则：

- content 主要提供结构和 token 对齐位置。
- style_context 是 content 查询 style value 后得到的风格条件。
- 默认 concat cross-attention 让一个 content token 直接访问所有 ref 的 style token。
- ref 数增加不一定单调变好，需要看训练时 ref 分布、注意力分散和 style 一致性。
