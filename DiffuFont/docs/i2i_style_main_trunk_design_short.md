# I2I 字体风格迁移：精简版设计

## 1. 总体原则

- **content 是主干**，决定字形结构
- **style 是条件**，提供局部写法
- **patch-level** 负责融合 content 与 style，输出 semantic tokens
- **pixel-level / PiT** 只吃 semantic tokens + timestep，不直接吃 style tokens

---

## 2. Style Encoder

### 目标
输出一组**无强位置、可被主干查询的 style prototypes**，表示局部写法原型，而不是整字结构。

### 输入
- 8 张 style 图

### 结构
1. **CNN backbone**
   - 4 个 stage
   - 3×3 Conv + Norm + GELU
   - 只做 2 次下采样
   - **不加显式位置编码**

2. **转成 tokens**
   - feature map flatten
   - 线性投影到主干维度 `D`

3. **prototype 压缩**
   - 用 `K` 个 learnable queries 做 attention pooling
   - 输出：
     - `P ∈ R^{B × K × D}`

### 含义
每个 prototype 表示一类局部写法原型，例如：
- 边缘质感
- 收笔方式
- 粗细变化
- 转角风格
- 墨迹纹理

### 设计原则
- **不用强位置编码**
- **不用深 Transformer 主干**
- **重点保留局部写法，不强调整字结构**

---

## 3. Patch-level 主干

### 输入
- content patch tokens 作为唯一主流

### 层数分配（以 12 层为例）

#### 前 4 层：只学内容结构
- Self-Attention
- FFN

作用：
- 建立字形结构
- 建立 patch 间关系
- 不注入 style

#### 中间 4 层：注入 style
每层做：
1. Self-Attention
2. Cross-Attention 到 style prototypes
3. FFN

公式：
- `R = CrossAttn(Q = Norm(H), K = P, V = P)`
- `H = H + λ_s * R`

其中：
- `H` 是 content 主流
- `P` 是 style encoder 输出的 prototypes

作用：
- 由 content 决定当前位置需要什么写法
- 从 style memory 中读取风格信息
- 残差写回 content 主流

#### 后 4 层：只做主干整理
- Self-Attention
- FFN

作用：
- 传播已经吸收的风格信息
- 整理成稳定的 semantic tokens

---

## 4. 输出到 PiT

主干最后输出：
- `s_N` = semantic tokens

然后构造：
- `s_cond = s_N + t`

再送入 PiT。

### PiT 的职责
- 不直接查 style
- 只根据 `s_cond` 做逐像素调制与细化

---

## 5. 核心逻辑

### content 主干负责
- 字形结构
- 全局上下文
- 最终 semantic tokens

### style encoder 负责
- 局部写法原型
- 风格记忆库

### cross-attention 负责
- 让 content patch 按需查询 style
- 把写法信息写回主干

### PiT 负责
- 把 patch 级语义展开成像素级细化

---

## 6. 最终结论

这套设计的核心是：

- **content 一开始就进主干**
- **style 不做并行主干，只做可查询 memory**
- **style 只在中间层注入**
- **最后由 content 主流沉淀出 semantic tokens**
- **PiT 只依赖 semantic tokens + timestep**
