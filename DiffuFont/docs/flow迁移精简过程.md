# 当前项目改成 Flow 的精简过程

本文档给出一条**最小改造路径**：在尽量复用现有代码的前提下，把当前 `Stage C: latent diffusion` 改成 `Flow Matching / Rectified Flow`。  
目标不是重写整套模型，而是**只替换主训练范式与采样方式**。

基于当前架构，以下模块可以直接保留：

- `Stage A: VAE 预训练`
- `Stage B: Style 预训练`
- `Content Encoder`
- `Style Encoder`
- `global_style_pool`
- `local_style_pool`
- `style_proj`
- `style_global_proj`
- `GlyphDiTBlock` 的条件注入结构

原因是当前主干已经是一个标准的**conditional latent transformer**：输入包含 `x_t`、`t`、`content_tokens`、`style_tokens`、`style_global`，这与 Flow 模型所需接口天然兼容。当前 latent 空间是 `B x 4 x 16 x 16`，内容条件是 `256` 个 token，风格条件包括 `style_global` 与 `style_tokens`，这些都无需推倒重做。

---

## 1. 改造原则

只改下面四件事：

1. 把 `Stage C` 的目标从 diffusion 噪声预测改成 flow / velocity 预测  
2. 把训练样本构造从 `q(x_t | x_0)` 改成线性插值 `x_t = (1-t)x_0 + tx_1`  
3. 把推理从 diffusion sampler 改成 ODE 积分  
4. 重新调少量训练超参

其余模块先不动。

---

## 2. 当前结构里哪些不动

### 保留不变

- `VAE`
  - 继续把 `1 x 128 x 128` 压到 `4 x 16 x 16 latent`
- `Content Encoder`
  - 继续输出 `content_tokens in R^(B x 256 x 512)`
- `Style Encoder`
  - 继续输出 `local_tokens` 与 `global_tokens`
- `Global + Static Local Style Pool`
  - 继续输出 `style_global` 与 `style_tokens`
- `DiT Backbone`
  - 继续保留：
    - `self-attention`
    - `content cross-attention`
    - `style cross-attention`
    - `AdaLN(style_global)`

### 只改语义，不改接口

当前模型输出如果是：

```python
pred = model(x_t, t, content_tokens, style_tokens, style_global)
```

那么改成 Flow 后，这个接口继续保留，只是 `pred` 的含义从：

- `pred_noise` / `pred_v`

变成：

- `pred_flow` / `pred_velocity`

---

## 3. Flow 版训练目标

### 定义两个端点

在当前任务里，可以定义：

- `x1 = z_target`：目标字形经过 VAE encoder 后的 latent
- `x0 ~ N(0, I)`：标准高斯噪声

其中 `x1` 就是当前 diffusion 训练里的目标 latent。

### 采样时间

```python
t ~ Uniform(0, 1)
```

### 构造中间状态

最简版本直接使用线性插值：

```python
x_t = (1 - t) * x0 + t * x1
```

### 监督目标

最小版本直接预测常量速度场：

```python
u_t = x1 - x0
```

训练目标：

```python
L_flow = || model(x_t, t, cond) - (x1 - x0) ||^2
```

这就是最容易落地的 **basic flow matching** 版本。

---

## 4. 训练循环怎么改

## 原 diffusion 逻辑

通常是：

```python
z = vae.encode(target)
eps = randn_like(z)
t = sample_timesteps(...)
x_t = q_sample(z, t, noise=eps)
pred = model(x_t, t, cond)
loss = diffusion_loss(pred, eps or v)
```

## 改成 flow 后

替换为：

```python
z1 = vae.encode(target).latent  # x1
z0 = torch.randn_like(z1)       # x0

t = torch.rand(B, device=z1.device)
t_view = t.view(B, 1, 1, 1)

x_t = (1.0 - t_view) * z0 + t_view * z1
target_flow = z1 - z0

pred_flow = model(
    x_t,
    t,
    content_tokens=content_tokens,
    style_tokens=style_tokens,
    style_global=style_global,
)

loss_flow = F.mse_loss(pred_flow, target_flow)
```

如果你当前 `Stage C` 还有附加损失，例如：

- style contrastive loss
- reconstruction-related auxiliary loss
- 其他 regularization

建议第一版迁移时：

- **先只保留主 flow loss**
- 其他损失后续再逐个加回

这样最容易判断迁移是否成功。

---

## 5. 推理怎么改

Flow 模型的推理不再是 diffusion 去噪，而是从随机噪声出发，对 ODE 做数值积分。

### 初始状态

```python
x = torch.randn(B, 4, 16, 16, device=device)
```

### ODE 形式

```python
dx/dt = model(x, t, cond)
```

### 最简 Euler 采样

```python
x = x0
for i in range(num_steps):
    t = i / num_steps
    dt = 1.0 / num_steps
    v = model(x, t, cond)
    x = x + dt * v

z_pred = x
img = vae.decode(z_pred)
```

### 推荐实践

第一版先用：

- `Euler`
- `num_steps = 16 / 24 / 32`

等模型跑通后，再尝试：

- `Heun`
- `RK2`
- 自适应 ODE solver

---

## 6. 代码层面的最小改造清单

### 6.1 数据部分

**不用改** dataset。  
当前 batch 里的这些仍然照常使用：

- `content glyph`
- `target glyph`
- `style refs`
- `style_ref_mask`

---

### 6.2 VAE

**不用改**。

只要仍能提供：

```python
z_target = vae.encode(target_glyph)
recon = vae.decode(z_pred)
```

即可。

---

### 6.3 内容与风格编码

**不用改**。

继续保留：

```python
content_tokens = content_encoder(content_img)
style_tokens, style_global = style_system(style_refs, style_ref_mask)
```

---

### 6.4 DiT 主干

只检查两件事：

#### A. timestep embedding 是否支持 `t in [0,1]`

如果你当前 time embedding 是按 diffusion 的整数 step 写的，例如：

```python
t in {0, 1, ..., 999}
```

那么改成：

```python
t in [0, 1]
```

常见做法：

- 直接对 `t` 做 sinusoidal embedding
- 或先映射到某个尺度再做 MLP

#### B. 输出头是否仍是 latent shape

确保模型输出仍为：

```python
B x 4 x 16 x 16
```

只是语义变成 velocity。

---

### 6.5 loss 模块

新增一个 `flow_loss()`，替换原 diffusion loss：

```python
def flow_matching_loss(model, z1, cond):
    z0 = torch.randn_like(z1)
    t = torch.rand(z1.size(0), device=z1.device)
    t_view = t[:, None, None, None]

    x_t = (1 - t_view) * z0 + t_view * z1
    target = z1 - z0

    pred = model(x_t, t, **cond)
    return F.mse_loss(pred, target)
```

---

### 6.6 sampler 模块

新增一个 `flow_sample()`，替换原 `ddim_sample()` / `p_sample_loop()`：

```python
def flow_sample(model, cond, shape, steps=24):
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps

    for i in range(steps):
        t = torch.full((shape[0],), i / steps, device=device)
        v = model(x, t, **cond)
        x = x + dt * v

    return x
```

采样得到的 latent 再 decode：

```python
z = flow_sample(...)
img = vae.decode(z)
```

---

## 7. 训练策略建议

## 第一阶段：先跑通最小版本

建议先这样：

- 保留 `Stage A`
- 保留 `Stage B`
- `Stage C` 只用 `flow loss`
- 不加对比损失
- 不加复杂采样器
- 不做 rectification / reflow

目标是先验证：

1. loss 是否正常下降
2. decode 后图像是否成字
3. content 是否对齐
4. style 是否基本保真

---

## 第二阶段：再加回已有技巧

等最小版稳定后，再逐项恢复：

1. style branch 局部解冻策略  
2. style contrastive auxiliary loss  
3. 更好的 solver  
4. self-conditioning / rectified tricks

这样最容易定位问题来源。

---

## 8. 训练超参需要重点重调的地方

迁移成 Flow 后，最可能需要重调的是：

- 学习率
- warmup 步数
- `style_unfreeze_step`
- `style_residual_gate`
- batch size
- t 的采样分布
- 采样步数

### 建议初始值

可以先从下面起步：

- `lr`: 先沿用当前值，再准备下调到 `5e-5`
- `t`: 先 `Uniform(0,1)`
- `solver`: 先 `Euler`
- `steps`: 先 `24`
- `style branch`: 先沿用当前冻结策略

---

## 9. 最容易踩坑的点

### 1. 时间嵌入没改干净

如果模型还按 diffusion 的整数 step 理解时间，而训练传入的是 `[0,1]` 浮点数，性能会直接受影响。

### 2. 输出目标尺度不稳定

`x1 - x0` 的幅度可能和原 diffusion target 不同，早期训练可能波动更大。必要时可以：

- 对 latent 做标准化
- 对 loss 做简单 rescale

### 3. 附加损失一次加太多

第一次迁移不要同时保留太多 auxiliary 项，否则很难判断主问题出在哪。

### 4. 把“Rectified Flow”理解成必须一开始就上复杂版本

不是。第一步先做 **basic flow matching**。  
只要这一步能稳定生成，再考虑 rectification / reflow。

---

## 10. 建议的实施顺序

### 第 1 步

复制当前 `Stage C` 训练脚本，新建：

- `train_flow.py`
- `losses/flow_matching.py`
- `sampler/flow_sampler.py`

### 第 2 步

在 `train_flow.py` 中替换：

- diffusion forward process
- diffusion loss
- diffusion sampler

### 第 3 步

保留原条件路径：

- `content_tokens`
- `style_tokens`
- `style_global`

不改 backbone block 结构。

### 第 4 步

先用小规模实验验证：

- loss 曲线
- 可视化生成结果
- 不同步数下的采样效果

### 第 5 步

确认基本可用后，再决定要不要：

- 加回辅助损失
- 改成更强的 solver
- 做真正的 rectified flow 增强

---

## 11. 一句话版本

对你当前项目来说，**改成 Flow 的最优路径不是重构全模型，而是把 `Stage C` 从 latent diffusion 换成 flow matching，并保留现有 VAE、content/style encoder 和条件化 DiT 主干不动**。

---

## 12. 可直接参考的最小伪代码

```python
# encode conditions
content_tokens = content_encoder(content_img)
style_tokens, style_global = style_module(style_refs, style_ref_mask)

# target latent
z1 = vae.encode(target_img).latent

# flow matching sample
z0 = torch.randn_like(z1)
t = torch.rand(z1.size(0), device=z1.device)
tv = t[:, None, None, None]

x_t = (1 - tv) * z0 + tv * z1
target_flow = z1 - z0

pred_flow = model(
    x_t,
    t,
    content_tokens=content_tokens,
    style_tokens=style_tokens,
    style_global=style_global,
)

loss = F.mse_loss(pred_flow, target_flow)
loss.backward()
optimizer.step()
```

推理：

```python
x = torch.randn(B, 4, 16, 16, device=device)
steps = 24
dt = 1.0 / steps

for i in range(steps):
    t = torch.full((B,), i / steps, device=device)
    v = model(
        x,
        t,
        content_tokens=content_tokens,
        style_tokens=style_tokens,
        style_global=style_global,
    )
    x = x + dt * v

pred_img = vae.decode(x)
```

---

## 13. 验收标准

迁移成功至少满足下面四条：

1. `loss_flow` 稳定下降  
2. `vae.decode(z_pred)` 输出可辨认字形  
3. 内容字与目标字符一致  
4. 风格能够从参考字形中继承，而不是退化成平均字体

如果只满足前两条，不满足后两条，说明 backbone 虽然学会了 transport，但条件利用还不够，需要再调 style/content 条件路径，而不是否定 Flow 方案本身。
