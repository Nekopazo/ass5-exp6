# DiffuFont 模型结构说明（Teacher-Only / Style-Driven）

## 1. 总体结论

当前主干是标准 diffusion U-Net，条件分支只保留 style。

- `baseline`：无条件 token
- `part_only`：style token 条件
- `style_only`：style token 条件

`part_style` 与 student distillation 已移除。

## 2. 条件数据流

```mermaid
flowchart TD
    X0[Target x0 256x256] -->|resize| XL[x0_latent 1x128x128]
    XT[x_t + timestep t] --> U

    C[content_img 1x128x128] --> CE[ContentEncoder]
    CE --> CR[content residuals]

    S[style refs BxRx1x128x128] --> SE[StyleEncoder CNN]
    SE --> PT[patch tokens Bx(R*P)xD]
    PT --> TL[TokenLayer: learnable queries cross-attn]
    TL --> ST[style tokens BxTxD]

    XL --> U[UNet backbone]
    CR --> U
    ST --> U
    U --> Y[eps_hat / v_hat 1x128x128]
```

## 3. Style 分支

- 输入：`style_img`
  - 支持 `(B,1,128,128)` 单参考图
  - 支持 `(B,R,1,128,128)` 多参考图
- StyleEncoder：4 层 stride-2 卷积（输出 8x8 feature map）
- Patch 展平：`B x (R*P) x D`
- TokenLayer：`T` 个 learnable query 与 patch 做 cross-attention
- 输出：`style_tokens = (B,T,D)`，默认 `T=8, D=256`

## 4. 训练入口

`train.py` 仅 teacher 训练，不含 student/KD。

## 5. 预训练入口

`pretrain_style_encoder.py` 对 style 分支做预训练：

- 双视图（R1/R2）
- bucket 采样 reference
- 增强：crop + mask + affine + resize
- Loss：InfoNCE + Consistency + Token Diversity

输出 checkpoint 可直接通过 `--pretrained-style-encoder` 导入主模型。
