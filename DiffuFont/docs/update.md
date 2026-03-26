# 阶段 1：最小 DiP 改造方案

## 目标

在**尽量不改现有 DiT 主干**的前提下，先验证一件事：

> 在字体生成任务里，给普通 DiT 末端增加一个轻量的 **Patch Detailer Head**，是否能明显提升局部细节质量。

这一阶段严格参考 DiP 的核心思路：

- 主干 DiT 负责 **global structure**
- 末端 detailer 负责 **local detail refinement**
- 采用 **post-hoc refinement**，不修改主干内部结构

---

## 现有模型

当前主干：

- 12 层普通 DiT
- 前 8 层注入 content（cross-attention）
- 后 6 层注入 style（modulation）
- 当前已经可以生成字形的大致结构

阶段 1 不改这些机制，只在最后加一个细节头。

---

## 改造原则

### 保留不变
- 现有 12 层 DiT 主干
- content cross-attention 逻辑
- style modulation 逻辑
- 原有 diffusion / flow 训练框架

### 新增
- 一个 **Patch Detailer Head**
- 放在 DiT backbone 最后输出之后
- 输入为：
  - 主干最后一层的 patch feature
  - 对应位置的 noisy patch
- 输出为：
  - 每个 patch 的噪声预测或残差预测

---

## 整体流程

```text
noisy font image x_t
    │
    ├─ patchify(P=16)
    │
    ▼
DiT Backbone（保持原结构）
    ├─ 前部 content cross-attn
    ├─ 后部 style modulation
    └─ 输出 S_global ∈ [B, N, D]
    │
    ▼
对每个 patch i:
    ├─ 取 global feature s_i
    ├─ 取 noisy patch p_i
    └─ 输入 Patch Detailer Head
           ↓
         输出 ε_i
    │
    ▼
reassemble all ε_i
    │
    ▼
full-image prediction