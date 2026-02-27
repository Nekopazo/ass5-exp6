# DiffuFont Teacher-Student 两段式改造方案（Stage-A 四线）

## 1. 目标

1. 保留当前线路并扩展成两段式训练（Teacher -> Student）。
2. Stage-A 第一阶段做四线实验：`baseline` / `part_only` / `style_only` / `part_style`。
3. `parts` 仅作为训练辅助，不成为部署依赖。
4. Student 训练与推理都只用单张 `style_img`。

---

## 2. Stage-A 第一阶段四线定义

| 线路 | 条件输入 | 目的 |
|---|---|---|
| `baseline` | `content` | 下界，对照组 |
| `part_only` | `content + parts` | 复用当前主线，验证 parts 上界能力 |
| `style_only` | `content + style_img` | 对齐 Student 输入域 |
| `part_style` | `content + style_img + optional parts` | 主推荐 Teacher 线路 |

说明：

- 你要的“当前线路 + style_img 组成第一部分”，就是把当前 `part_only` 保留，同时新增 `style_only` 和 `part_style`，并和 `baseline` 组成四线。
- 后续 Stage-B Student 只蒸馏/继承 `style_only` 能力，不接 parts。

---

## 3. style_img 应该加在哪里

结论：`style_img` 不走 content 分支，走独立 style 编码分支，最终作为 `encoder_hidden_states[2]` 注入现有 style cross-attention。

### 3.1 模型注入位（按现有代码）

当前 UNet 的 style 条件注入位已经存在：

- `MCADownBlock2D`: `self_attn -> style_cross_attn`
- `UNetMidMCABlock2D`: `self_attn -> style_cross_attn`
- `StyleUpBlock2D`: `self_attn -> style_cross_attn`

这些模块都从 `encoder_hidden_states[2]` 读取 style tokens。  
因此改造重点是“构造 tokens”，不是重写 UNet 主干。

### 3.2 style token 放置策略

新增 `style_img_encoder`，输入 `style_img (B,1,256,256)`，输出 `style_tokens (B,M,D)`。  
推荐 `M=8, D=256`，与当前 `cross_attention_dim=256` 保持一致。

然后统一设置：

- `baseline`: `encoder_hidden_states[2] = None`
- `part_only`: `encoder_hidden_states[2] = part_tokens`
- `style_only`: `encoder_hidden_states[2] = style_tokens`
- `part_style`: `encoder_hidden_states[2] = fused_tokens`

### 3.3 `part_style` 融合方式（推荐先用门控加和）

保持 token 数不变，最少改动：

`fused_tokens = LN(style_tokens + g * part_tokens)`

其中：

- `g = sigmoid(MLP([pool(style_tokens), pool(part_tokens), has_parts]))`
- 无 parts 或被 part-drop 时，强制 `g=0`，退化为 `style_tokens`

优点：

- 不改 cross-attn 维度，不改 block 结构。
- 保持 `style_only` 与 `part_style` 参数可复用。
- 更容易保证“parts 可有可无”。

---

## 4. attention 怎么做

不建议新开第二套 attention，直接复用现有 style cross-attn 即可：

1. Query：各层 UNet 特征图（现有实现）。
2. Key/Value：`encoder_hidden_states[2]`（四线分别提供不同 tokens）。
3. 顺序：保持当前块内顺序  
   `ResBlock -> Self-Attn -> Cross-Attn(style_tokens) -> ResBlock`。
4. 分辨率：先沿用当前 `attn_scales=16,32`，避免早期高分辨率过强风格注入。

---

## 5. 两段式训练

## Stage-A（Teacher 预训练，四线）

通用主损失：

- `L_diff`: diffusion / flow matching 主损失

仅含 parts 的线可加：

- `L_nce_part`：沿用现有双视图 InfoNCE（`part_only` / `part_style`）

仅 `part_style` 推荐加：

- `L_cons`：同 batch 有/无 parts 的一致性约束

`L_cons = MSE(eps_with_parts, eps_no_parts.detach())`

`part_style` 总损失建议：

`L = L_diff + λ_nce * L_nce_part + λ_cons * L_cons`

## Stage-B（Student 蒸馏，style-only）

- Student 输入：`content + style_img`
- Teacher 固定为 Stage-A 的 `part_style` 最优权重
- 蒸馏：
  - `L_diff_s`（对 GT）
  - `L_kd`（对 Teacher 输出）

`L_student = L_diff_s + λ_kd * L_kd`

---

## 6. 文件级改造清单（按当前仓库）

## 6.1 `dataset.py`

- 常驻返回 `style_img`（同字体不同字符，单张）。
- `parts` 改为可选，缺失不抛错。
- 增加 `has_parts` 标志。

## 6.2 `train.py`

- 增加 `--stage {teacher,student}`。
- 增加 `--teacher-line {baseline,part_only,style_only,part_style}`（Stage-A 用）。
- 增加 `--teacher-ckpt`（Stage-B 用）。
- 增加 `--part-drop-prob`、`--lambda-cons`、`--lambda-kd`。

## 6.3 `models/source_part_ref_unet.py`

- 新增 `style_img_encoder`。
- 新增 `encode_style_tokens(style_img)`。
- 复用现有 `encode_part_tokens(part_imgs, part_mask)`。
- 新增 `fuse_style_part_tokens(...)`。
- 前向统一输出 `encoder_hidden_states[2]` 给现有 cross-attn。

## 6.4 `models/model.py`

- `TeacherTrainer` 支持四线模式切换。
- `StudentDistillTrainer` 只跑 style-only 输入。

## 6.5 `inference.py`

- Student 推理接口只保留 `content + style_img`。
- 完全去掉 parts 推理依赖。

---

## 7. 关于“Teacher 要不要只用 parts”

在这个目标下，不推荐 Teacher 只用 parts。  
更稳的做法是：Stage-A 用四线对照，最终 Teacher 选 `part_style`，Student 固定 `style_only`。

理由：

1. `style_only` 是 Student 的真实输入域，Teacher 需要覆盖这一路径。
2. `part_style` + part-drop + 一致性约束，才能让 parts 成为增益项而不是硬依赖。
3. 这样最终部署可直接用 Student（单 style 图推理）。

---

## 8. 最小执行顺序

1. 先改 `dataset.py`：补齐 `style_img` 常驻、`parts` 可选。
2. 再改模型：补 `style_img_encoder` 与 token 融合。
3. 跑 Stage-A 四线。
4. 从 `part_style` 中选最优 Teacher 做 Stage-B 蒸馏。
5. 完成 style-only Student 推理回归测试。
