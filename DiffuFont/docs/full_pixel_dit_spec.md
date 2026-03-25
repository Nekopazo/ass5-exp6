# Full Pixel DiT（Content + Multi-Style）重构架构规范

## 1. 概述

本模型为 **Full Pixel DiT 双层结构**：

- **Patch-level（前 12 层）**：融合 content / style / global / timestep，生成 semantic tokens
- **Pixel-level（后 2 层）**：仅接收 noisy image 的 pixel tokens 与 patch-level 传下来的 `s_cond`，执行像素级细节建模

关键原则：

1. `content / style / global` 条件在 patch-level 完成融合
2. pixel-level 不再直接访问 `content / style / global` 原始条件
3. `timestep` 不作为独立支路输入 PiT，而是先并入 semantic tokens
4. `s_cond` 是 patch-level 与 pixel-level 之间的唯一条件桥梁

---

## 2. 输入与符号

### 输入

- `content image`: `[B, 1, 128, 128]`
- `style images`: `[B, K, 1, 128, 128]`，`K = 8`
- `noisy target`: `x_t: [B, 1, 128, 128]`
- `timestep`: `t: [B]`

### 超参数

- patch size = `16`
- patch 数 = `64`（8 x 8）
- 每 patch pixel = `256`
- patch hidden dim = `D`
- pixel hidden dim = `D_pix`

---

## 3. 编码器

### 3.1 Content Encoder（空间对齐）

```text
content -> Conv(stride=16) -> c_tok: [B, 64, D]
```

- 与 target patch 严格对齐
- 每个 token 对应同一空间位置

### 3.2 Style Token Encoder（局部风格）

```text
style images -> encoder -> s_tok: [B, K, 64, D]
```

- 不要求与 target patch 一一空间对齐
- 每张 style 图单独保留
- 作为 patch-level style cross-attention 的 memory

### 3.3 Style Global Encoder（全局风格）

```text
style images -> CNN / encoder -> pooling -> s_g: [B, D]
```

- 用于提供整体风格偏置
- 可在主训练阶段冻结
- 这是字体任务下允许保留的任务特化分支，不要求强行改成论文的 single-stage 设定

### 3.4 Timestep Embedding

```text
t -> embedding -> t_emb: [B, D]
```

### 3.5 全局条件融合

```text
[t_emb ; s_g] -> MLP -> g: [B, D]
```

`g` 仅用于 patch-level AdaLN 调制。

---

## 4. Patch-level DiT（12 层）

### 4.1 Patch Embedding

```text
x_t -> patchify -> x_patch: [B, 64, D]
```

### 4.2 Content 注入（对齐融合）

```text
[x_patch ; c_tok] -> Linear -> [B, 64, D]
```

### 4.3 Patch Block 结构

每层：

```text
x = x + SelfAttn(AdaLN(x, g))
x = x + StyleModule(x, s_tok)      # 仅后 6 层启用
x = x + FFN(AdaLN(x, g))
```

### 4.4 Block 1 ~ 6（结构阶段）

仅执行：

- self-attn
- AdaLN(g)
- FFN

### 4.5 Block 7 ~ 12（风格阶段）

对每一张 style 图分别做 cross-attention：

```text
y_k = CrossAttn(x, s_tok[:, k])
```

堆叠后：

```text
y_all: [B, 64, K, D]
```

### 4.6 Style 融合（逐 reference 可学习加权）

```text
w = Linear(y_all)
w = softmax(w, dim=K)
y = sum_k w_k * y_k
x = x + y
```

### 4.7 Patch-level 输出与传递

```text
s_sem: [B, 64, D]
s_cond = s_sem + t_emb[:, None, :]
```

- `s_sem` 是 patch-level 融合 content / style / global 后得到的语义 token
- `t_emb` 在进入 pixel-level 前加到每个 patch 的 semantic token 上
- pixel-level 接收的是 `s_cond`，而不是单独的 `timestep` 分支

---

## 5. Pixel-level PiT（2 层）

### 核心原则

pixel-level 不接收 `content / style / global` 原始条件，仅接收：

- 当前 noisy image 的 pixel tokens
- patch-level 传下来的 `s_cond`

这对应论文里的逻辑：pixel-level 的条件来自 semantic tokens，而 `timestep` 通过 `s_cond = s_sem + t_emb` 间接注入，而不是作为新的原始输入分支再进入 PiT。

### 5.1 Pixel Embedding

```text
x_t -> per-pixel Linear / Conv1x1 -> [B, 128, 128, D_pix]
reshape by patch(16x16) -> x_pix: [B, 64, 256, D_pix]
```

- 每个 patch 内有 `256` 个 pixel tokens
- 后续 PiT 计算按 patch 维度展开

### 5.2 Pixel-wise AdaLN 条件化

对每个 patch 的 semantic token `s_cond` 做逐像素条件展开：

```text
s_cond: [B, 64, D]
-> reshape -> [B*64, D]
-> Phi(Linear)
-> [B*64, 256, 6*D_pix]
-> split -> (beta1, gamma1, alpha1, beta2, gamma2, alpha2)
```

每个张量形状均为：

```text
[B*64, 256, D_pix]
```

含义：

- 同一个 patch 内的 `256` 个 pixel，不共享一套 AdaLN 参数
- modulation 是逐像素的，不是 patch-wise broadcast
- 这是论文里 pixel-level 与 patch-level 的关键区别之一

### 5.3 PiT Block

单个 PiT block 由两项核心机制构成：

1. `pixel-wise AdaLN`
2. `token compaction`

#### Step A: 按 patch 展平

```text
x_pix: [B, 64, 256, D_pix]
-> x: [B*64, 256, D_pix]
```

#### Step B: Compaction

对每个 patch 内的 `256` 个 pixel tokens 做压缩：

```text
C: [256, D_pix] -> [D]
```

得到：

```text
x_cmp: [B, 64, D]
```

这里的 compaction 只为降低全局 attention 的计算量，不是 VAE 式的信息瓶颈。

#### Step C: 全局 Self-Attention

```text
x_cmp -> global MHSA over 64 patches -> [B, 64, D]
```

attention 在压缩后的 patch tokens 上做，用于建模 patch 间全局语义交互。

#### Step D: Expand

将 attention 输出重新展开回 patch 内 pixel tokens：

```text
E: [D] -> [256, D_pix]
```

得到：

```text
attn_update: [B*64, 256, D_pix]
```

#### Step E: Pixel-wise AdaLN + 残差更新

记当前 pixel tokens 为 `x`，则单层 PiT 的更新写成：

```text
x = x + alpha1 ⊙ AttnUpdate(gamma1 ⊙ RMSNorm(x) + beta1)
x = x + alpha2 ⊙ FFN(gamma2 ⊙ RMSNorm(x) + beta2)
```

其中：

- `AttnUpdate(...)` 对应 `compaction -> global self-attn -> expand`
- `(beta*, gamma*, alpha*)` 全部来自该 patch 的 `s_cond`
- 每个 pixel 都有自己独立的一组调制参数

### 5.4 关键约束

- 每个 patch 的 pixel 仅使用该 patch 对应的 `s_cond`
- PiT 不直接再看 `content / style / global`
- `timestep` 不单独输入 PiT，而是已经并入 `s_cond`

---

## 6. 输出头

```text
PiT output
-> reshape
-> Conv1x1 / Linear head
-> pred_v: [B, 1, 128, 128]
```

- 输出预测的是 pixel-space velocity / flow
- 不是直接重建 clean image

---

## 7. 训练

采用 pixel-space Rectified Flow 目标：

```text
x1 = target image
x0 ~ N(0, I)
t ~ Uniform(0, 1)
x_t = (1 - t) * x0 + t * x1
v_t = x1 - x0
pred_v = model(x_t, t, content, style)
loss_rf = MSE(pred_v, v_t)
```

规范默认以 `loss_rf` 为主训练目标。

如需为字体任务增加 `L1 / perceptual` 辅助项，它们只能是 auxiliary loss，不能替代 Rectified Flow 主目标。

---

## 8. 核心总结

- content：对齐注入到 patch-level
- style：逐 reference 做 cross-attn，再做加权融合
- global：保留任务特化全局风格分支，只在 patch-level 提供全局偏置
- `s_cond = s_sem + t_emb`：patch-level 到 pixel-level 的唯一条件桥梁
- pixel-level：使用 pixel-wise AdaLN 和 token compaction
- training：采用 pixel-space Rectified Flow velocity matching

---

## 9. 一句话总结

本模型先在 patch-level 融合 content / style / global 得到 `s_sem`，再构造 `s_cond = s_sem + t_emb` 传入 PiT；PiT 用 pixel-wise AdaLN 和 token compaction 在 pixel space 中执行 Rectified Flow 预测。
