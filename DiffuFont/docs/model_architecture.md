# FontDiffusionUNet 结构图（支持全局/部件开关）

## 开关说明
- `--use-global-style`:
  - 开：`style_img -> style_encoder -> style_feats`
  - 关：`style_feats` 用全零特征占位
- `--use-part-style`:
  - 开：`part_imgs(+part_mask)` 或 `style_img` 输入 `PartStyleEncoder`，得到 `part_style_vec` 并融合到 `style_feats`
  - 关：跳过部件分支

## 整体结构
```mermaid
flowchart TD
    XT[x_t noisy glyph] --> DOWN
    T[timestep t] --> SIN[Sinusoidal Embedding]
    SIN --> TMLP[Time MLP -> t_emb]

    CIMG[content_img] --> CENC[Content Encoder]
    CENC --> CFEATS[content_feats multi-scale]

    SIMG[style_img] --> GSW{use_global_style?}
    GSW -->|ON| SENC[Style Encoder]
    GSW -->|OFF| SZERO[Zero style feats]
    SENC --> SRAW[style_feats raw]
    SZERO --> SRAW

    PIMG[part_imgs] --> PSW{use_part_style?}
    PMASK[part_mask] --> PSW
    SIMG --> PSW
    PSW -->|ON| PENC[PartStyleEncoder]
    PENC --> PVEC[part_style_vec]
    PSW -->|OFF| PNONE[None]

    SRAW --> PFUSE[Part Fusion\npart_to_style + part_gate(t_emb)\nselected scales]
    PVEC --> PFUSE
    TMLP --> PFUSE
    PNONE --> PFUSE
    PFUSE --> SFEATS[style_feats fused]

    subgraph UNet["Diffusion U-Net Backbone"]
        DOWN[Down Blocks + DACA + AdaLN]
        DOWN --> BTM[Bottom ResBlock]
        BTM --> DRES[DecoderResBlock + FGSA (+AttnX optional)]
        DRES --> UP[Up Blocks + FGSA (+AttnX optional) + AdaLN]
        UP --> OUT[Out Conv]
    end

    CFEATS --> DOWN
    TMLP --> DOWN
    SFEATS --> DRES
    SFEATS --> UP
    OUT --> X0[x0_hat]
```

## 训练与推理共用流程
- 训练：`train_step -> model.forward -> encode_conditions -> forward_with_feats`
- 推理：`ddim_sample` 先调用一次 `encode_conditions`，然后每个 DDIM 步复用条件特征跑 `forward_with_feats`
