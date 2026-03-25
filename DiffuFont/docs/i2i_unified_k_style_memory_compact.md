# I2I 字体风格迁移（统一 K 个 Style Memory 方案）

## 1. 设计目标

本方案用于汉字图生图风格迁移，核心目标是：

- **content 决定字形结构**
- **style memory 同时承载整体风格与局部写法**
- **不显式拆分 global / local**
- **K 个 memory 优先理解为容量，不强行要求分工**
- **patch-level 主干负责融合 content 与 style**
- **PiT / pixel-level 只接收 semantic tokens + timestep**

---

## 2. 总体结构

整体分成三部分：

1. **Style Encoder**
   - 输入 8 张 style 图
   - 输出 dense style tokens
   - 再压缩成 `K` 个 unified style memory tokens

2. **Patch-level 主干**
   - content 一开始直接进入主干
   - 前 4 层只学习内容结构
   - 中间 4 层查询 style memory
   - 后 4 层整理为 semantic tokens

3. **PiT / pixel-level**
   - 只接收 semantic tokens 和 timestep
   - 不直接接收 style memory

---

## 3. Style Encoder

## 3.1 输入
- 8 张 style 图
- 分辨率：`128 × 128`

## 3.2 CNN Backbone
使用中等深度 CNN，做两次下采样：

\[
128 \rightarrow 64 \rightarrow 32
\]

最终输出 feature map：

\[
F \in \mathbb{R}^{B \times C \times 32 \times 32}
\]

说明：
- **不使用显式位置编码**
- CNN 负责提取局部笔触、边缘、粗细、收笔、纹理等写法信息
- 不要求 style encoder 学习整字结构语义

## 3.3 Dense Style Tokens
将 feature map 展平并投影到主干维度 `D`：

\[
S \in \mathbb{R}^{B \times 1024 \times D}
\]

8 张 style 图拼接后：

\[
S_{all} \in \mathbb{R}^{B \times 8192 \times D}
\]

---

## 4. Unified Style Memory 压缩

不拆分 global 与 local，直接从 dense style tokens 压缩出 `K` 个 style memory。

## 4.1 压缩方式
使用普通软加权压缩：

\[
A = \text{softmax}(W_a S_{all})
\]

其中：

\[
A \in \mathbb{R}^{B \times 8192 \times K}
\]

然后对 dense style tokens 加权求和：

\[
P_k = \sum_i A_{ik} S_{all,i}
\]

得到：

\[
P \in \mathbb{R}^{B \times K \times D}
\]

其中：
- `P` 就是统一 style memory
- 不显式区分哪个 token 是 global、哪个 token 是 local
- 所有 memory token 共同承载整体风格与局部写法

## 4.2 K 的含义
- `K` 优先理解为 **容量**
- 不强行要求每个 memory token 有明确职责
- 不做强分工设计

推荐：
- `K = 4` 作为最小 baseline
- 后续可对比 `K = 8`、`K = 16`

---

## 5. Patch-level 主干

## 5.1 输入
content 图一开始直接进入主干：

\[
H \in \mathbb{R}^{B \times L \times D}
\]

其中：
- `L` 为 content patch token 数
- `D` 为主干隐藏维度

---

## 5.2 层数安排（12 层示例）

### 前 4 层：只学内容结构
每层只做：
- Self-Attention
- FFN

作用：
- 建立字形结构
- 建立 patch 间关系
- 不注入 style

### 中间 4 层：查询 style memory
每层做：

1. Self-Attention
2. 多头 Cross-Attention 查询 style memory
3. FFN

公式：

\[
R_{sty} = \text{MHA}(Q=\text{Norm}(H), K=P, V=P)
\]

\[
H \leftarrow H + \lambda_s R_{sty}
\]

其中：
- `H` 是 content 主流
- `P` 是统一 style memory
- `\lambda_s` 为 style 注入系数

作用：
- content 决定当前位置需要什么风格/写法
- 从统一 style memory 中读取信息
- 将风格信息残差写回 content 主流

### 后 4 层：整理语义
每层只做：
- Self-Attention
- FFN

作用：
- 传播已吸收的风格信息
- 整理成稳定的 semantic tokens

---

## 6. semantic tokens 与 PiT

patch-level 主干最后输出：

\[
s_N
\]

这就是 semantic tokens。

然后构造：

\[
s_{cond} = s_N + t
\]

其中：
- `t` 为 timestep embedding

再将 `s_{cond}` 输入 PiT。

### PiT 的职责
- 不直接看 style memory
- 只根据 semantic tokens 做逐像素调制与细化

---

## 7. 方案特点

### 优点
- 结构统一，不需要显式拆分 global / local
- `K` 更自然地表示 style memory 容量
- 对简单字体和复杂字体都适用
- 主干逻辑清晰：content 主流 + 中间层查 style memory
- PiT 继续保持干净，只接收 semantic tokens

### 设计原则
- **content 始终是主流**
- **style 只作为 memory 被查询**
- **K 是容量，不是预定义分工数**
- **整体风格与局部写法统一存放在 style memory 中**
- **最终都沉淀到 semantic tokens**

---

## 8. 最终结论

本方案最终采用：

- **Style Encoder：32×32 单尺度 CNN**
- **无显式位置编码**
- **从 dense style tokens 直接压缩出 K 个 unified style memory tokens**
- **不显式拆分 global / local**
- **content 一开始直接进入主干**
- **前 4 层不注入 style**
- **中间 4 层通过多头 cross-attention 查询 unified style memory**
- **后 4 层整理成 semantic tokens**
- **最终由 PiT 只接收 semantic tokens + timestep**
