# 极简流程图（看这个就够）

```mermaid
flowchart TD
    A[输入: content_img + style_img + 可选part_imgs] --> B[encode_conditions]

    B --> C{use_global_style?}
    C -->|开| D[StyleEncoder提取全局style_feats]
    C -->|关| E[style_feats置零]

    B --> F{use_part_style?}
    F -->|开| G[PartStyleEncoder提取part_style_vec]
    F -->|关| H[跳过part分支]

    D --> I[Part Fusion: 用part_style_vec调制style_feats]
    E --> I
    G --> I
    H --> I

    I --> J[forward_with_feats]
    J --> K[Down: DACA + AdaLN]
    K --> L[Bottom ResBlock]
    L --> M[Up: FGSA(+AttnX) + AdaLN]
    M --> N[Out Conv]
    N --> O[x0_hat 预测字形]
```

## 一句话理解
- `global style`：管“整体字风格”。
- `part style`：管“局部部件风格”，再融合进整体风格特征。
