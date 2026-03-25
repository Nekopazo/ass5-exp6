# Style Memory 压缩改动（仅这一项）

## 改动目标

不加额外 loss，不强行做分工，只通过结构设计降低 `K` 个 style memory 全部塌成同一个平均值的风险。

---

## 原方案

原来使用一个共享打分头，对所有 memory 一起打分：

\[
A = \text{softmax}(W_a S_{all})
\]

其中：

- \(S_{all} \in \mathbb{R}^{B \times M \times D}\)：dense style tokens
- \(W_a \in \mathbb{R}^{D \times K}\)

然后压缩得到：

\[
P_k = \sum_i A_{ik} S_{all,i}
\]

得到：

\[
P \in \mathbb{R}^{B \times K \times D}
\]

---

## 修改后方案

将共享打分头改为 **K 个独立打分头**。

对第 \(k\) 个 memory，单独计算打分：

\[
a^{(k)} = \text{softmax}(S_{all} w_k + b_k)
\]

其中：

- \(w_k \in \mathbb{R}^{D \times 1}\)
- \(b_k \in \mathbb{R}\)

然后分别压缩得到第 \(k\) 个 memory：

\[
P_k = \sum_i a_i^{(k)} S_{all,i}
\]

最后拼接得到：

\[
P \in \mathbb{R}^{B \times K \times D}
\]

---

## 直观理解

原方案中，所有 memory 共享同一个打分空间。  
修改后，每个 memory 都有自己的打分视角：

- memory 1 有自己的偏好
- memory 2 有自己的偏好
- ...
- memory K 有自己的偏好

这样做的作用不是强迫它们明确分工，而是让它们**不容易完全看成同一堆东西**。

---

## 这样做的好处

### 1. 更不容易全部塌成同一个 mean
每个 memory 自己决定更关注哪些 style tokens。

### 2. 不需要额外 loss
不需要 diversity loss、orthogonal loss、top-k 路由等额外约束。

### 3. 仍然符合“容量”设定
这里的 `K` 仍然表示 style memory 容量，不要求每个 memory 有明确职责。

### 4. 改动很小
只改 style memory 压缩器，不改主干结构。

---

## 可能的局限

### 1. 不能保证完全不塌
独立打分头只能降低坍塌风险，不能绝对保证所有 memory 都不同。

### 2. 仍可能出现部分相似
例如 4 个 memory 中有 2 个比较接近，这是允许的。

### 3. 这是“弱去重”，不是“强分工”
适合当前“把 K 当容量”的设计目标。

---

## 主干部分不变

patch-level 主干仍保持：

- content 一开始直接进入主干
- 前 4 层只学内容结构
- 中间 4 层用多头 cross-attention 查询 style memory
- 后 4 层整理成 semantic tokens
- PiT 只接收 semantic tokens + timestep

查询公式不变：

\[
R_{sty} = \text{MHA}(Q=\text{Norm}(H), K=P, V=P)
\]

\[
H \leftarrow H + \lambda_s R_{sty}
\]

---

## 最终结论

本次仅修改 style memory 压缩方式：

- **从共享打分头**
- 改为 **K 个独立打分头**

目的不是强迫 memory 分工，而是让 `K` 个 memory 更自然地从不同视角聚合 style tokens，从而降低全部塌成同一个平均值的风险。