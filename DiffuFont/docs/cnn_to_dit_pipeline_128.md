# CNN → DiT 内容注入流程（128 分辨率简明版）

## 🎯 目标
将 **content（结构信息，如 glyph / edge / mask）** 从 CNN 特征注入到 DiT token 中。

---

## 🧠 整体流程

```text
content image (128×128)
→ CNN encoder
→ feature map (B, Cc, 8, 8)
→ flatten → content tokens (B, 64, Cc)
→ Linear 投影 → (B, 64, Dc)
→ 与 DiT token concat
→ Linear 融合 → (B, 64, 512)
→ Self-Attention
→ MLP
```

---

## 📐 维度对齐（关键）

### DiT 主干
- 输入图像：128×128
- patch size：16
- token 网格：8×8
- token 数：64
- hidden dim：512

```text
x: [B, 64, 512]
```

---

### CNN 输出

```text
c_feat: [B, Cc, 8, 8]
```

---

### 转换为 token

```python
c = c_feat.flatten(2).transpose(1, 2)
# [B, Cc, 8, 8] → [B, 64, Cc]
```

---

## 🔗 融合方式（核心）

```python
# x: [B, 64, 512]
# c: [B, 64, Dc]

x = torch.cat([x, c], dim=-1)   # [B, 64, 512 + Dc]
x = fuse_proj(x)                # [B, 64, 512]
```

这一步对应 UNet 里的：
- 通道拼接
- 1×1 卷积压回主干维度

在 DiT 里，1×1 conv 对应的是 Linear。

---

## 🔁 DiT Block 内结构

```python
x = x + pos_embed          # DiT 自带位置编码
x = fuse(x, c)             # content 注入
x = x + attn(norm1(x))     # self-attention
x = x + mlp(norm2(x))      # MLP
```

---

## 📌 关键原则

### 1. token 必须一一对齐
```text
8×8 feature map ⇄ 64 tokens
```

### 2. content 是局部条件
不要做全局 broadcast：

```text
[B, 1, Dc] → broadcast 到 64
```

应该使用逐位置对齐的局部条件：

```text
[B, 64, Dc]
```

### 3. 不改变主干维度
concat 后必须再映射回 512，保持 DiT 主干结构不变。

### 4. 使用 DiT 自带位置编码即可
如果 content token 和 DiT token 一一对齐，就不需要额外再给 content 单独加一套位置编码。

---

## 🚀 一句话总结

**CNN 提取 8×8 的局部结构特征，展平成 64 个 content token，与 DiT 的 64 个 patch token 按位置拼接融合，再映射回 512 维后送入 self-attention。**
