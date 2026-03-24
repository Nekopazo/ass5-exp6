# Full Pixel DiT（Content + Multi-Style）最简架构规范

## 1. 概述

本模型为 **Full Pixel DiT 双层结构**：

- **Patch-level（前 12 层）**：融合 content / style / timestep → 生成 semantic tokens  
- **Pixel-level（后 2 层）**：仅使用 semantic tokens → 执行像素级细节生成  

关键原则：

1. 所有条件在 patch-level 完成融合  
2. pixel-level 不再直接访问任何原始条件（content / style / global / timestep）  
3. semantic tokens 是唯一的条件桥梁  

---

## 2. 输入与符号

### 输入

- `content image`: `[B, 1, 128, 128]`
- `style images`: `[B, K, 1, 128, 128]`（K=8）
- `noisy target`: `x_t: [B, 1, 128, 128]`
- `timestep`: `t: [B]`

---

### 超参数

- patch size = `16`
- patch 数 = `64`（8×8）
- 每 patch pixel = `256`
- hidden dim = `D`
- pixel dim = `D_pix`

---

## 3. 编码器

### 3.1 Content Encoder（对齐）

```
content → Conv(stride=16) → c_tok: [B, 64, D]
```

- 与 target patch **严格对齐**
- 每个 token 对应同一空间位置

---

### 3.2 Style Token Encoder（局部风格）

```
style images → encoder → s_tok: [B, K, 64, D]
```

- 不做空间对齐
- 每张 style 图单独保留
- 用于 patch-level cross-attn

---

### 3.3 Style Global Encoder（全局风格）

```
style images → CNN → pooling → s_g: [B, D]
```

- 轻量 CNN
- **主训练阶段冻结（frozen）**
- 提供整体风格偏置

---

### 3.4 Timestep Embedding

```
t → embedding → t_emb: [B, D]
```

---

### 3.5 全局条件融合

```
[t_emb ; s_g] → MLP → g: [B, D]
```

用于 patch-level AdaLN

---

## 4. Patch-level DiT（12 层）

### 4.1 Patch Embedding

```
x_t → patchify → x_patch: [B, 64, D]
```

---

### 4.2 Content 注入（对齐 concat）

```
[x_patch ; c_tok] → Linear → [B, 64, D]
```

---

## 4.3 Patch Block 结构

每层：

```
x = x + SelfAttn(AdaLN(x, g))
x = x + (Style Module，仅后6层)
x = x + FFN(AdaLN(x, g))
```

---

### 4.4 Block 1 ~ 6（结构阶段）

仅：
- self-attn
- AdaLN(g)
- FFN

---

### 4.5 Block 7 ~ 12（风格阶段）

对每一张 style 图：

```
y_k = CrossAttn(x, s_tok[:, k])
```

堆叠：

```
y_all: [B, 64, K, D]
```

---

### 4.6 Style 融合（可学习加权）

```
w = Linear(y_all)
w = softmax(w, dim=K)
y = Σ w * y_all
x = x + y
```

---

### 4.7 Patch-level 输出

```
s_sem: [B, 64, D]
```

---

## 5. Pixel-level PiT（2 层）

### 核心原则

pixel-level 不接收任何原始条件，仅使用 semantic tokens

---

## 5.1 Pixel Embedding

```
x_t → Conv1x1 → reshape → x_pix: [B,64,256,D_pix]
```

---

## 5.2 Pixel Block

步骤：

1. Compaction  
2. Self-Attention  
3. Expand  
4. Modulation（来自 s_sem）  
5. FFN 更新  

---

### 关键约束

每个 patch 的 pixel 仅使用该 patch 的 semantic token

---

## 6. 输出头

```
→ reshape → Conv1x1 → [B,1,128,128]
```

---

## 7. 训练

```
loss = MSE(pred, target)
```

---

## 8. 核心总结

- content：对齐注入  
- style：逐张 attention + 加权融合  
- global：冻结 CNN + AdaLN  
- semantic：唯一条件桥梁  
- pixel-level：纯执行层  

---

## 9. 一句话总结

本模型通过 patch-level 融合多源条件为 semantic tokens，在 pixel-level 通过 modulation 精细生成像素，实现 full pixel diffusion。
