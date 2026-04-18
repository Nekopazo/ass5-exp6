# 中文字风格迁移 Diffusion Transformer（增强版）

## 1. 任务定义
目标：在保留 **content 字形结构** 的前提下，将其迁移到目标 **style 风格**，通过 diffusion Transformer 生成清晰图像。

## 2. 总体结构
输入：
- x_content
- x_style
- z_t

流程：
1. CNN 编码 content / style → 16×16×256
2. content 作为 Query 查询 style（cross-attention）
3. 残差融合并拼接
4. 投影到主干维度
5. patchify z_t → Transformer（12层）
6. adaLN-Zero 注入
7. 输出 x_pred

## 3. 条件编码
Content: [B,1,128,128] → [B,16,16,256]  
Style:   [B,1,128,128] → [B,16,16,256]

## 4. Cross-Attention
F_c → [B,256,256]  
F_s → [B,256,256]  

Q = F_c, K = F_s, V = F_s  
F_cs = Attn(Q,K,V)

## 5. 条件融合
F_fuse = F_c + F_cs  
F_cat = concat(F_c, F_fuse)  
F_cond = Linear(512 → D)

## 6. Patch Embedding
z_t → patches (8×8) → dim=64  
64 → 128 → D  

## 7. Transformer（12层）
结构：
- RMSNorm
- SwiGLU
- RoPE
- qk-norm（可选）
- adaLN-Zero

每层：
T → RMSNorm → scale/shift → SA → gate → residual  
T → RMSNorm → scale/shift → FFN(SwiGLU) → gate → residual  

## 8. 输出
T → Linear → patch → 图像

## 9. 训练
z_t = t x + (1-t) ε  
x_pred = net(z_t)  
v_pred = (x_pred - z_t)/(1-t)  
loss = ||v_pred - v||^2

## 10. 核心原则
- x-pred 是关键
- content 控结构
- style 控外观
- patch size 控细节
- bottleneck 提升泛化
