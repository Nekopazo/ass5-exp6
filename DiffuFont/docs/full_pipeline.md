# DiffuFont 完整流程文档

## 1. 项目总览

DiffuFont 是一个基于扩散模型的中文字体生成系统，采用 **Teacher-Student 两阶段训练**：

- **Stage-A (Teacher)**：使用 `content + style_img + parts`（可选组合）训练扩散模型
- **Stage-B (Student)**：仅使用 `content + style_img` 蒸馏训练，去除 parts 依赖

核心设计：**parts 仅作为训练辅助**，不成为部署依赖。最终 Student 推理只需一张 style 参考图。

---

## 2. 数据源与格式

### 2.1 原始数据文件

| 数据文件 | 路径 | 格式 | 说明 |
|---|---|---|---|
| 字符列表 | `CharacterData/CharList.json` | JSON 字符串数组 | 训练使用的所有汉字列表 |
| 字体列表 | `DataPreparation/FontList.json` | JSON 字符串数组 | 字体文件名列表（取 stem） |
| 内容字体 LMDB | `DataPreparation/LMDB/ContentFont.lmdb` | LMDB | key=`ContentFont@字` → value=PNG bytes |
| 训练字体 LMDB | `DataPreparation/LMDB/TrainFont.lmdb` | LMDB | key=`字体名@字` → value=PNG bytes |
| PartBank LMDB | `DataPreparation/LMDB/PartBank.lmdb` | LMDB | key=`DataPreparation/PartBank/字体名/part_NNN_UXXXX.png` → value=PNG bytes |

### 2.2 LMDB Key 格式

```
ContentFont.lmdb   →  "ContentFont@永"          → 灰度 PNG 图像 bytes
TrainFont.lmdb     →  "SimSun@永"               → 灰度 PNG 图像 bytes
PartBank.lmdb      →  "DataPreparation/PartBank/SimSun/part_001_U6C38.png"  → 灰度 PNG 图像 bytes
```

### 2.3 数据量

- **样本总数** = `字体数 × 字符数`（如 50 字体 × 3000 字 = 150,000 样本）
- 每个样本从同字体随机采 **1~8 个 part 图像**（两组独立采样用于 InfoNCE）
- 每个样本随机选择 **同字体不同字** 作为 style 参考图

---

## 3. 数据处理流水线

### 3.1 Dataset (`FontImageDataset`)

**初始化阶段**：

1. 加载 `CharList.json` → `self.char_list: List[str]`
2. 加载 `FontList.json` → `self.font_list_stems: List[str]`
3. 打开 3 个 LMDB 环境（`content_env`, `train_env`, `part_env`），保持持久的只读事务
4. 全量扫描 `TrainFont.lmdb` 的 key，提取所有字体名
5. 对每个字体，遍历 `char_list` 检查 content + train 两个 LMDB 中是否同时存在
6. 构建 `self.samples: List[Tuple[字体名, 字符索引]]`
7. 若启用 PartBank：扫描 LMDB key 构建 `part_bank_by_font: Dict[字体名, List[行]]`，并过滤掉没有 part 的字体

**`__getitem__(index)`** 返回一个样本 dict：

```python
{
    "font": str,                    # 字体名，如 "SimSun"
    "char": str,                    # 字符，如 "永"
    "content": Tensor (1, 256, 256),  # 内容字形（标准字体）
    "input": Tensor (1, 256, 256),    # 目标字形（训练字体，作为 GT）
    "has_parts": float,             # 0.0 或 1.0
    # ---- 以下为可选 ----
    "style_img": Tensor (1, 256, 256),  # 同字体不同字的参考图
    "style_char": str,
    "parts": Tensor (P_a, 1, 64, 64),   # PartBank 视图 A（1~8 个 part）
    "parts_b": Tensor (P_b, 1, 64, 64), # PartBank 视图 B（独立采样，用于 InfoNCE）
    "part_mask": Tensor (P_a,),          # 全 1
    "part_mask_b": Tensor (P_b,),        # 全 1
}
```

### 3.2 图像预处理

| 图像类型 | 原始格式 | 处理流程 | 输出尺寸 | 像素范围 |
|---|---|---|---|---|
| content / target / style_img | LMDB PNG → PIL "L" (灰度) | `ToTensor()` → `Normalize(0.5, 0.5)` | `(1, 256, 256)` | `[-1, 1]` |
| part 图像 | LMDB PNG → PIL "L" → resize | 手动 `/255 → *2 - 1` | `(1, 64, 64)` | `[-1, 1]` |

### 3.3 Collate（批量拼接）

`collate_fn()` 将 batch 内变长的 parts 填充为统一长度：

```python
{
    "content": (B, 1, 256, 256),
    "target":  (B, 1, 256, 256),
    "font_ids": (B,),                    # batch 内整数字体 ID（用于 InfoNCE）
    "has_parts": (B,),                   # 0.0/1.0
    "style_img": (B, 1, 256, 256),       # 可选
    "parts":    (B, P_max, 1, 64, 64),   # 零填充到 batch 内最大 part 数
    "part_mask":  (B, P_max),            # 有效位=1，填充位=0
    "parts_b":  (B, P_max_b, 1, 64, 64),
    "part_mask_b": (B, P_max_b),
}
```

---

## 4. 模型架构

### 4.1 顶层模型 (`SourcePartRefUNet`)

```
输入:
  x_t_latent: (B, 1, 128, 128)   ← 从 256×256 双线性下采样
  t:          (B,)                ← 时间步索引
  content_img:(B, 1, 256, 256)   ← 像素空间
  style_img:  (B, 1, 256, 256)   ← 可选
  part_imgs:  (B, P, 1, 64, 64)  ← 可选
  part_mask:  (B, P)              ← 可选

输出:
  eps_hat / v_hat: (B, 1, 128, 128)   ← 噪声预测(Diffusion) 或速度预测(Flow Matching)
```

### 4.2 Latent 空间（伪 latent）

**没有使用 VAE**。采用双线性插值作为 pixel ↔ latent 转换：

```
encode_to_latent:   (B,1,256,256) → bilinear ↓2 → (B,1,128,128)
decode_from_latent: (B,1,128,128) → bilinear ↑2 → (B,1,256,256)
```

UNet 在 128×128 分辨率上运行，降低计算量。

### 4.3 Content Encoder

```
ContentEncoder (在 256×256 全分辨率运行)
  ├── DBlock (SNConv + AvgPool2d) × 6 层
  └── 输出多尺度残差特征:
      [x256, c128, c64, c32, c16, c8]

传入 UNet 的 Down Block 特征（取 index 1~4）:
  Down-0 (128→64): c128  (64ch)
  Down-1 (64→32):  c64   (128ch)
  Down-2 (32→16):  c32   (256ch)
  Down-3 (16→16):  c16   (512ch)
```

Content 特征通过 **ChannelAttnBlock** 在每个 Down Block 中以 channel attention 方式注入。

### 4.4 Style 条件编码 — 四种 Conditioning Profile

| 模式 | `encoder_hidden_states[2]` | 说明 |
|---|---|---|
| `baseline` | `None` | 无风格条件（对照组） |
| `part_only` | `part_tokens (B, M, D)` | 仅 parts |
| `style_only` | `style_tokens (B, M, D)` | 仅 style 参考图 |
| `part_style` | `fused_tokens (B, M, D)` | style + 门控 parts 融合 |

#### Part Token 编码路径

```
part_imgs (B, P, 1, 64, 64)
  → reshape → (B*P, 1, 64, 64)
  → part_patch_encoder (3层 Conv+BN+SiLU + AdaptiveAvgPool + Linear)
  → (B*P, 256) → reshape → (B, P, 256)
  → masked mean pool (DeepSets) + LayerNorm
  → (B, 256)
  → style_token_mlp (Linear→SiLU→Linear)
  → reshape → (B, M=8, D=256)  ← part_tokens
```

#### Style Image Token 编码路径

```
style_img (B, 1, 256, 256)
  → style_img_encoder (3层 Conv+BN+SiLU + AdaptiveAvgPool + Linear)
  → (B, 256) → LayerNorm
  → style_img_token_mlp (Linear→SiLU→Linear)
  → reshape → (B, M=8, D=256)  ← style_tokens
```

#### Part-Style 融合（门控加和）

```
style_pool = mean(style_tokens, dim=1)   → (B, 256)
part_pool  = mean(part_tokens, dim=1)    → (B, 256)
gate_in    = cat([style_pool, part_pool, has_parts])  → (B, 513)
gate       = sigmoid(MLP(gate_in)) * has_parts        → (B, 1, 1)  [0~1 标量]

fused_tokens = LayerNorm(style_tokens + gate * part_tokens)
```

- 当无 parts 或被 drop 时，`gate=0`，退化为 `style_tokens`
- 设计目的：parts 是**增益项**而非硬依赖

#### Contrastive 投影头（InfoNCE 用）

```
part_tokens (B, M, D)
  → mean(dim=1) → (B, D)
  → contrastive_head (LN→Linear→SiLU→Linear)
  → L2 normalize → z (B, D)
```

### 4.5 UNet 主干

```
UNet (sample_size=128, in_channels=1, out_channels=1)
  ├── conv_in:  (1→64) 3×3
  ├── time_proj + time_embedding → temb
  │
  ├── Down Blocks (4 × MCADownBlock2D):
  │   Block-0: 128×128 → 64×64,   64ch → 64ch
  │   Block-1:  64×64  → 32×32,   64ch → 128ch
  │   Block-2:  32×32  → 16×16,  128ch → 256ch    ← style attn 启用 (attn_scales=16,32)
  │   Block-3:  16×16  → 16×16,  256ch → 512ch    ← style attn 启用
  │
  ├── Mid Block (UNetMidMCABlock2D):
  │   16×16, 512ch → 512ch                          ← style attn 启用, content attn 关闭
  │
  ├── Up Blocks:
  │   Block-0 (StyleUpBlock2D): 16×16 →  32×32, 512ch → 256ch  ← style attn 启用(16)
  │   Block-1 (StyleUpBlock2D): 32×32 →  64×64, 256ch → 128ch  ← style attn 启用(32)
  │   Block-2 (UpBlock2D):      64×64 → 128×128, 128ch → 64ch  ← 无 style attn
  │   Block-3 (UpBlock2D):     128×128,          64ch →  64ch   ← 无 style attn
  │
  ├── conv_norm_out (GroupNorm) + SiLU
  └── conv_out: (64→1) 3×3
```

**每个 MCADownBlock2D 内部流程**（per layer）：

```
input
  → ChannelAttnBlock(concat(hidden, content_feat)) → content 注入
  → ResnetBlock2D(+temb)
  → SelfAttention → StyleCrossAttention(K/V = style_tokens) → 风格注入
  → ResnetBlock2D(+temb)
output → (skip connection 保存)
  → Downsample2D (stride-2 conv)
```

**Style Cross-Attention** 的 K/V 来自 `encoder_hidden_states[2]`（即上文的 part_tokens / style_tokens / fused_tokens），Q 来自 UNet 特征图。

**Style Attention 仅在 32×32 和 16×16 分辨率启用**（`--attn-scales 16,32`），低分辨率避免过强风格注入。

### 4.6 参数规模

| 组件 | 通道配置 | 说明 |
|---|---|---|
| `content_encoder` | 64→128→256→512→1024→2048 (6层 DBlock) | 全分辨率 256×256 |
| `unet` | (64, 128, 256, 512) 4层 + Mid 512 | 128×128 运行 |
| `part_patch_encoder` | 1→64→128→256→GAP→Linear(256) | 64×64 输入 |
| `style_img_encoder` | 1→64→128→256→GAP→Linear(256) | 256×256 输入 |
| `style_token_mlp` | 256→1024→2048 | 单向量→8 个 token |
| `contrastive_head` | LN + 256→256→256 | 仅训练时 |
| `style_part_gate` | 513→256→1 | part_style 融合门控 |

默认 token 参数：`M=8 tokens, D=256 dim`，`cross_attention_dim=256`。

---

## 5. 训练流程

### 5.1 两阶段设计

#### Stage-A (Teacher) — 四线实验

| 线路 | `--teacher-line` | 条件输入 | 额外损失 |
|---|---|---|---|
| baseline | `baseline` | content only | L_diff |
| part_only | `part_only` | content + parts | L_diff + L_nce |
| style_only | `style_only` | content + style_img | L_diff |
| **part_style** | `part_style` | content + style_img + parts | L_diff + L_nce + L_cons |

推荐 Teacher 线路：`part_style`。

#### Stage-B (Student) — 蒸馏

- Student 固定使用 `style_only` 模式：`content + style_img`
- 加载 Stage-A `part_style` 的最优 Teacher 权重
- Teacher 用 `part_style` 模式生成 KD target

```bash
python train.py --stage student \
    --teacher-ckpt checkpoints/part_style/ckpt_best.pt \
    --teacher-distill-mode part_style \
    --lambda-kd 1.0
```

### 5.2 Diffusion Trainer 的 `train_step`

```
1. 加载 batch → x0 (target), content, style_img, part_imgs, part_mask

2. CFG Dropout (p=0.1):
   - 随机选 10% 样本，将 content/style_img/part_imgs 全部设为 1.0 (白色=null)
   - part_mask 设为 0.0

3. Part Drop (part_style 模式, p=part_drop_prob):
   - 随机选部分样本，仅清空 part_imgs/part_mask（保留 style_img）
   - 让模型学会"style_img alone 也能工作"

4. Pixel → Latent:
   x0_latent = bilinear(x0, 128×128)   # 无梯度

5. 前向扩散加噪:
   t ~ Uniform(0, T-1)
   x_t, eps = NoiseScheduler.add_noise(x0_latent, t)

6. 噪声预测:
   eps_hat = model(x_t, t, content, style_img, part_imgs, part_mask)

7. 计算损失（见下文 §6）

8. 梯度累积 + AdamW 更新:
   loss / grad_accum → backward
   每 grad_accum 步执行一次 optimizer.step()
   OneCycleLR.step()（每 optimizer step 一次）

9. 自动 checkpoint / sampling / logging
```

### 5.3 Flow Matching Trainer 的 `train_step`

```
与 Diffusion Trainer 类似，关键区别:

4. Latent 空间线性插值:
   t ~ Uniform(0, 1)
   x1 = randn_like(x0_latent)      # 纯噪声
   x_t = (1-t) * x0_latent + t * x1
   v_target = x1 - x0_latent        # 速度目标

5. 速度预测:
   t_idx = round(t * (T-1))          # 映射到离散时间步
   v_hat = model(x_t, t_idx, content, ...)

6. 损失:
   L_fm = MSE(v_hat, v_target)      # 替代 L_mse
```

### 5.4 优化器与学习率

| 配置 | 值 |
|---|---|
| 优化器 | AdamW, weight_decay=1e-4 |
| 学习率 | `--lr 2e-4` (max_lr) |
| LR Scheduler | OneCycleLR |
| 预热 | 前 5% 步数（pct_start=0.05） |
| 退火策略 | cosine |
| 初始 LR | max_lr / 25 = 8e-6 |
| 最终 LR | 初始 LR / 1000 ≈ 8e-9 |
| total_steps | `epochs * steps_per_epoch / grad_accum` |

### 5.5 混合精度 (AMP)

- `--precision fp32`（默认）—— 不启用
- `--precision bf16` —— 需 Ampere+ GPU（sm_80+），autocast + bfloat16
- `--precision fp16` —— autocast + float16 + GradScaler

---

## 6. 损失函数

### 6.1 总损失

**Diffusion Trainer (Teacher stage, part_style 模式)**:

$$L = \lambda_{mse} \cdot L_{mse} + \lambda_{nce}(t) \cdot L_{nce} + \lambda_{cons} \cdot L_{cons}$$

**Flow Matching Trainer**:

$$L = \lambda_{fm} \cdot L_{fm} + \lambda_{nce}(t) \cdot L_{nce} + \lambda_{cons} \cdot L_{cons}$$

**Student stage** 额外增加:

$$L_{student} = L_{diff/fm} + \lambda_{kd} \cdot L_{kd}$$

### 6.2 各损失详解

#### L_mse — 噪声预测 MSE

$$L_{mse} = \text{MSE}(\hat{\epsilon}, \epsilon)$$

在 128×128 latent 空间计算。`λ_mse` 默认 1.0。

#### L_fm — 速度预测 MSE（Flow Matching）

$$L_{fm} = \text{MSE}(\hat{v}, v^*)$$

其中 $v^* = x_1 - x_0$（latent 空间）。`λ_fm` 默认 1.0。

#### L_nce — InfoNCE 对比损失

对**同一字体**的两组独立采样 part 集合 (parts_a, parts_b) 计算：

```
z_a = model.encode_contrastive_z(parts_a, mask_a)   → (B, D), L2 normalized
z_b = model.encode_contrastive_z(parts_b, mask_b)   → (B, D), L2 normalized
```

拼接为 `(2B, D)`，计算 cosine 相似度矩阵 / temperature (τ=0.07)：

$$L_{nce} = -\frac{1}{|P(i)|} \sum_{j \in P(i)} \log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}$$

- **正对**：同一字体的样本对（包括跨视图、batch 内同字体不同样本）
- **负对**：不同字体的样本对
- **目的**：让 part encoder 学到字体级别的风格表示

带 **线性 warmup**：前 `nce_warmup_steps`（默认 5000）步内从 0 线性增长到 `λ_nce=0.05`。

#### L_cons — 一致性约束

仅在 `part_style` 模式下有效。**核心设计：让 parts 不会过度偏离 style-only 的输出**。

```python
eps_with_parts = model(x_t, t, content, style_img, part_imgs, mode="part_style")
eps_no_parts   = model(x_t, t, content, style_img, None,      mode="style_only")
L_cons = MSE(eps_with_parts, eps_no_parts.detach())  # 仅排除 CFG drop 的样本
```

- `eps_no_parts.detach()`：style-only 输出作为 **anchor**（不更新）
- 梯度只更新 `eps_with_parts` 分支
- **效果**：惩罚 parts 让模型输出偏离 style-only 太远，确保 Student 蒸馏时 gap 最小

#### L_kd — 知识蒸馏损失

仅在 Student stage 使用：

```python
with torch.no_grad():
    eps_teacher = teacher_model(x_t, t, content, style_img, parts, mode="part_style")
L_kd = MSE(eps_hat_student, eps_teacher)
```

Student 用 `style_only` 输入，Teacher 用 `part_style` 输入，强制 Student 不借助 parts 也能逼近 Teacher 输出。

### 6.3 默认超参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `lambda_mse` / `lambda_fm` | 1.0 | 主损失权重 |
| `lambda_nce` | 0.05 | InfoNCE 权重 |
| `nce_warmup_steps` | 5000 | NCE warmup 步数 |
| `lambda_cons` | 0.0 | 一致性约束（需手动开启） |
| `lambda_kd` | 0.0 | 蒸馏损失（Student stage 开启） |
| `cfg_drop_prob` | 0.1 | CFG Dropout 概率 |
| `part_drop_prob` | 0.0 | Part Drop 概率（需手动开启） |

---

## 7. 推理 / 采样

### 7.1 Diffusion Trainer — DPM-Solver++ 采样

```
1. x_T ~ N(0, I) in latent (B, 1, 128, 128)
2. DPMSolverMultistepScheduler (dpmsolver++, 20 步):
   for t in dpm.timesteps:
       eps_cond = model(x_t, t, content, style_img, parts)
       if use_cfg:
           eps_uncond = model(x_t, t, 白色null, None, None, mode="baseline")
           eps = eps_uncond + gs * (eps_cond - eps_uncond)    # gs=7.5
       x_{t-1} = dpm.step(eps, t, x_t)
3. pixel = bilinear(x_0, 256×256).clamp(-1, 1)
```

### 7.2 Flow Matching Trainer — Euler ODE 采样

```
1. x_1 ~ N(0, I) in latent (B, 1, 128, 128)
2. 欧拉前向积分 (n_steps 步, dt = 1/n_steps):
   for i in range(n_steps):
       t = 1.0 - i * dt
       v_cond = model(x_t, round(t*(T-1)), content, style_img, parts)
       if use_cfg:
           v_uncond = model(x_t, ..., 白色null, None, None, mode="baseline")
           v = v_uncond + gs * (v_cond - v_uncond)
       x_t = x_t - dt * v
3. pixel = bilinear(x_0, 256×256).clamp(-1, 1)
```

### 7.3 CFG（Classifier-Free Guidance）

- 训练时以 10% 概率将所有条件（content + style + parts）置为白色/零
- 推理时同时跑有条件和无条件前向，用 guidance_scale 线性外推
- 早期训练（<5000 步）：自动关闭 CFG 采样（gs=1.0），避免噪声放大

---

## 8. Checkpoint 与保存

### 8.1 Checkpoint 内容

```python
{
    "model_state": OrderedDict,       # 完整模型权重
    "opt_state":   OrderedDict,       # AdamW 状态
    "step":        int,               # global_step
    "epoch":       int,
    "local_step":  int,
    "lr_schedule_state": dict,        # OneCycleLR 状态
    "diffusion_steps":   int,
    "total_steps":       int,
    "precision":         str,
    "grad_scaler_state": dict,        # 仅 fp16 时
}
```

### 8.2 组件拆分保存

开启 `--split-save-components` 后，额外保存：

```
checkpoints/components/
  ├── ckpt_step_5000.main_model.pt           # content_encoder + unet 权重
  └── ckpt_step_5000.trainable_vector_cnn.pt # part/style encoder + MLP + gate + contrastive head
```

### 8.3 日志

- `checkpoints/train_step_metrics.jsonl`：每 `--log-every-steps` 步写一行 JSON
- `checkpoints/samples/`：每 `--sample-every-steps` 步保存 content | GT | generated 对比图
- `checkpoints/train_run_config.json`：完整训练配置

---

## 9. 完整数据流图

```
┌─────────────────────────────── Dataset ──────────────────────────────────┐
│                                                                          │
│  ContentFont.lmdb ─── PNG → PIL "L" → ToTensor + Normalize(0.5,0.5)    │
│       key: "ContentFont@字"                                              │
│       → content (1, 256, 256) [-1,1]                                    │
│                                                                          │
│  TrainFont.lmdb ──── PNG → PIL "L" → ToTensor + Normalize(0.5,0.5)     │
│       key: "字体名@字"                                                   │
│       → target (1, 256, 256) [-1,1]     目标 GT                         │
│       → style_img (1, 256, 256) [-1,1]  同字体不同字                     │
│                                                                          │
│  PartBank.lmdb ──── PNG → PIL "L" → resize(64) → /255 → *2-1           │
│       key: ".../字体名/part_NNN_UXXXX.png"                              │
│       → parts (P, 1, 64, 64) [-1,1]    随机 1~8 个                      │
│       → parts_b (P', 1, 64, 64) [-1,1] 独立二次采样                     │
│                                                                          │
└──────────── collate_fn: pad parts → batch ───────────────────────────────┘
                              │
                              ▼
┌──────────────────────── Trainer.train_step ───────────────────────────────┐
│                                                                          │
│  1. CFG Dropout (p=10%): 随机置 null                                     │
│  2. Part Drop (可选): 随机清空 parts                                      │
│  3. target → bilinear ↓2 → x0_latent (1,128,128)                        │
│  4. t ~ U(0,T); add_noise → x_t_latent                                  │
│                                                                          │
│  5. model.forward:                                                       │
│     ┌─ ContentEncoder(content 256×256) → [c128, c64, c32, c16]          │
│     ├─ part_patch_encoder(parts 64×64) → DeepSets pool → MLP            │
│     │     → part_tokens (B, 8, 256)                                     │
│     ├─ style_img_encoder(style 256×256) → MLP                           │
│     │     → style_tokens (B, 8, 256)                                    │
│     ├─ fuse: style + gate * part → fused_tokens (B, 8, 256)             │
│     │                                                                    │
│     └─ UNet(x_t_latent 128×128, temb, [None, content_feats, tokens])    │
│          Down-0(128→64) → Down-1(64→32) → Down-2(32→16) → Down-3(16)   │
│          Mid(16) → Up-0(16→32) → Up-1(32→64) → Up-2(64→128) → Up-3     │
│          → eps_hat (1, 128, 128)                                         │
│                                                                          │
│  6. Loss:                                                                │
│     L = L_mse(eps_hat, eps) + λ_nce·L_nce + λ_cons·L_cons [+ λ_kd·L_kd]│
│                                                                          │
│  7. Gradient accumulation → AdamW step → OneCycleLR step                 │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────── Inference ────────────────────────────────────┐
│                                                                          │
│  z ~ N(0,I) (1, 128, 128)                                               │
│  DPM-Solver++ (20步) / Euler ODE (20步):                                 │
│    每步: eps/v = model(z_t, t, content, style_img, [parts])              │
│    + optional CFG (gs=7.5)                                               │
│  → z_0 (1, 128, 128) → bilinear ↑2 → pixel (1, 256, 256) [-1,1]       │
│                                                                          │
│  Student 推理: 仅需 content + style_img（无 parts）                       │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 10. 训练命令示例

### Stage-A: Teacher (part_style 主线)

```bash
python train.py \
    --data-root . \
    --stage teacher \
    --teacher-line part_style \
    --trainer diffusion \
    --epochs 50 --batch 16 --grad-accum 2 \
    --lr 2e-4 --precision bf16 \
    --lambda-nce 0.05 --nce-warmup-steps 5000 \
    --lambda-cons 0.1 --part-drop-prob 0.2 \
    --cfg-drop-prob 0.1 \
    --attn-scales 16,32 \
    --part-set-min 1 --part-set-max 8 \
    --save-dir checkpoints/part_style \
    --save-every-steps 5000 --sample-every-steps 300
```

### Stage-B: Student (蒸馏)

```bash
python train.py \
    --data-root . \
    --stage student \
    --teacher-ckpt checkpoints/part_style/ckpt_step_100000.pt \
    --teacher-distill-mode part_style \
    --trainer diffusion \
    --epochs 30 --batch 16 --grad-accum 2 \
    --lr 1e-4 --precision bf16 \
    --lambda-kd 1.0 \
    --cfg-drop-prob 0.1 \
    --save-dir checkpoints/student
```

### 推理

```bash
python inference.py \
    --checkpoint checkpoints/student/ckpt_best.pt \
    --data-root . \
    --conditioning-profile style_only \
    --trainer diffusion \
    --inference-steps 20 --guidance-scale 7.5 \
    --num-fonts 4 --num-chars 6 \
    --output results/comparison.png
```
