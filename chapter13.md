# 第13章：注意力机制优化

注意力机制是Transformer架构的核心组件，但也是计算和内存的主要瓶颈。本章深入探讨注意力机制的优化技术，从算法层面的改进到系统层面的实现优化，涵盖Flash Attention、多查询注意力、稀疏注意力模式以及线性注意力等前沿技术。这些优化对于在边缘设备上高效部署大语言模型至关重要。

## 13.1 Flash Attention原理与实现

### 13.1.1 标准注意力的计算瓶颈

标准的缩放点积注意力（Scaled Dot-Product Attention）计算公式为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中，$Q, K, V \in \mathbb{R}^{N \times d}$，$N$是序列长度，$d$是特征维度。

**内存复杂度分析**：
- 存储注意力矩阵 $S = QK^T$ 需要 $O(N^2)$ 内存
- 对于长序列（如 $N = 2048$），这会产生 4M 个元素的中间矩阵
- 在多头注意力中，这个开销会乘以头数
- 以FP16存储，仅注意力矩阵就需要 8MB 内存

**计算复杂度细分**：
1. 矩阵乘法 $S = QK^T$：$2N^2d$ FLOPs
2. Softmax计算：约 $5N^2$ FLOPs（exp、sum、div操作）
3. 输出计算 $O = PV$：$2N^2d$ FLOPs
4. 总计：约 $4N^2d + 5N^2$ FLOPs

**带宽瓶颈**：
标准实现需要多次读写HBM（高带宽内存）：
1. 加载 $Q, K$ 计算 $S = QK^T$：读取 $2Nd$ 个元素
2. 写回 $S$ 到HBM：写入 $N^2$ 个元素
3. 读取 $S$ 计算 softmax：读取 $N^2$ 个元素
4. 写回 $P = \text{softmax}(S)$：写入 $N^2$ 个元素
5. 读取 $P, V$ 计算最终输出：读取 $N^2 + Nd$ 个元素

总的HBM访问量约为 $O(N^2d + Nd^2)$。

**硬件特性考虑**：
现代GPU的内存层次结构：
- SRAM（共享内存）：~100KB，带宽 ~19TB/s
- HBM（全局内存）：~40GB，带宽 ~1.5TB/s
- 带宽差距：约13倍

这意味着如果能将计算保持在SRAM中，理论上可获得超过10倍的性能提升。

### 13.1.2 Flash Attention的核心思想

Flash Attention通过**平铺（tiling）**和**重计算（recomputation）**来优化内存访问：

1. **分块计算**：将输入分成小块，使其能够完全放入SRAM（片上内存）
2. **融合操作**：在SRAM中完成矩阵乘法和softmax，避免中间结果写回HBM
3. **增量softmax**：使用在线算法计算softmax，无需存储完整的注意力矩阵

**关键洞察**：
- 标准实现的瓶颈不是计算，而是内存带宽
- 通过减少HBM访问次数，即使增加一些重复计算也能获得净收益
- 利用softmax的数学性质，可以分块计算并正确合并结果

**算法创新点**：
1. **在线softmax算法**：无需两遍扫描即可计算softmax
2. **平铺策略**：根据SRAM大小优化块尺寸
3. **数值稳定性**：通过动态调整数值范围避免溢出

### 13.1.3 分块算法详解

设块大小为 $B_r \times B_c$，将 $Q$ 分成 $T_r = \lceil N/B_r \rceil$ 块，$K, V$ 分成 $T_c = \lceil N/B_c \rceil$ 块。

**块划分策略**：
- $Q$ 矩阵：$Q = [Q_1; Q_2; ...; Q_{T_r}]$，每个 $Q_i \in \mathbb{R}^{B_r \times d}$
- $K$ 矩阵：$K = [K_1; K_2; ...; K_{T_c}]$，每个 $K_j \in \mathbb{R}^{B_c \times d}$
- $V$ 矩阵：$V = [V_1; V_2; ...; V_{T_c}]$，每个 $V_j \in \mathbb{R}^{B_c \times d}$

**外层循环**（对每个 $Q$ 块）：
```
对于 i = 1 到 T_r:
    加载 Q_i 到SRAM
    初始化: O_i = 0, l_i = 0, m_i = -∞
    
    内层循环（对每个 K,V 块）：
    对于 j = 1 到 T_c:
        加载 K_j, V_j 到SRAM
        计算 S_{ij} = Q_i K_j^T / √d_k  # 大小: B_r × B_c
        
        # 增量softmax更新
        m_{ij} = rowmax(S_{ij})         # 每行的最大值
        P_{ij} = exp(S_{ij} - m_{ij})   # 数值稳定的exp
        l_{ij} = rowsum(P_{ij})          # 每行的和
        
        # 更新运行统计
        m_i^{new} = max(m_i, m_{ij})
        l_i^{new} = exp(m_i - m_i^{new}) * l_i + exp(m_{ij} - m_i^{new}) * l_{ij}
        
        # 更新输出
        O_i = diag(exp(m_i - m_i^{new}))^{-1} * O_i + 
              diag(exp(m_{ij} - m_i^{new}))^{-1} * P_{ij} * V_j
              
        m_i = m_i^{new}, l_i = l_i^{new}
    
    写回 O_i 到HBM
```

**算法正确性保证**：
该算法正确实现了标准注意力的原因在于：
1. 每个输出块 $O_i$ 独立计算，对应于 $Q_i$ 与所有 $K, V$ 的注意力
2. 增量更新确保了跨块的softmax归一化正确性
3. 数值稳定性通过动态调整最大值 $m_i$ 来保证

### 13.1.4 增量Softmax的数学推导

关键在于正确地合并部分softmax结果。设已处理前 $j-1$ 块，现处理第 $j$ 块：

**推导基础**：
完整的softmax计算为：
$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

为了数值稳定性，通常减去最大值：
$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

**增量更新推导**：
设已处理的部分结果为：
- $m_i^{(j-1)}$：前 $j-1$ 块的行最大值
- $l_i^{(j-1)}$：前 $j-1$ 块的归一化因子（未归一化的softmax分母）
- $O_i^{(j-1)}$：前 $j-1$ 块的加权输出和

处理第 $j$ 块时：

**最大值更新**：
$$m_i^{(j)} = \max(m_i^{(j-1)}, m_{ij})$$

**归一化因子更新**：
需要将旧的归一化因子调整到新的数值范围：
$$l_i^{(j)} = e^{m_i^{(j-1)} - m_i^{(j)}} l_i^{(j-1)} + \sum_k e^{S_{ijk} - m_i^{(j)}}$$

这里：
- $e^{m_i^{(j-1)} - m_i^{(j)}}$ 是缩放因子，调整旧的累积和
- $\sum_k e^{S_{ijk} - m_i^{(j)}}$ 是新块的贡献

**输出更新**：
$$O_i^{(j)} = \frac{e^{m_i^{(j-1)} - m_i^{(j)}} l_i^{(j-1)}}{l_i^{(j)}} O_i^{(j-1)} + \frac{1}{l_i^{(j)}} \sum_k e^{S_{ijk} - m_i^{(j)}} V_{jk}$$

证明这等价于完整计算：
$$O_i = \sum_{j,k} \frac{e^{S_{ijk} - \max_l S_{il}}}{\sum_{j',k'} e^{S_{ij'k'} - \max_l S_{il}}} V_{jk}$$

### 13.1.5 内存和计算复杂度分析

**SRAM使用量详细分析**：
- 存储一个 $Q$ 块：$B_r \times d$ 个元素
- 存储一个 $K, V$ 块：$2 \times B_c \times d$ 个元素
- 存储中间结果 $S_{ij}$：$B_r \times B_c$ 个元素
- 存储统计量 $m_i, l_i$：$2 \times B_r$ 个元素
- 存储输出块 $O_i$：$B_r \times d$ 个元素
- 总计：$O((B_r + B_c) \times d + B_r \times B_c)$

**HBM访问量对比**：

标准注意力：
- 读取 $Q, K, V$：$3Nd$ 次
- 写入/读取 $S$：$2N^2$ 次
- 写入/读取 $P$：$2N^2$ 次
- 总计：$O(N^2 + Nd)$ 次HBM访问

Flash Attention：
- 读取 $Q$：每块读一次，总计 $Nd$ 次
- 读取 $K, V$：每个 $Q$ 块都要读所有 $K, V$，总计 $2T_r \times Nd = 2Nd$ 次
- 写回输出：$Nd$ 次
- 总计：$O(Nd)$ 次HBM访问

**带宽需求降低**：
- 标准实现：需要 $O(N^2)$ 的带宽
- Flash Attention：只需要 $O(Nd)$ 的带宽
- 改善因子：$O(N/d)$，对于长序列效果显著

**计算复杂度分析**：
- 每个 $(i,j)$ 块对：计算 $S_{ij} = Q_i K_j^T$ 需要 $2B_r B_c d$ FLOPs
- 总块对数：$T_r \times T_c = N^2/(B_r B_c)$
- 总FLOPs：$2N^2d$，与标准实现相同
- 额外的softmax更新计算：$O(N^2)$，相对较小

### 13.1.6 Flash Attention v2的改进

Flash Attention v2主要优化了算法的并行性和硬件利用率：

1. **改进的工作分配**：
   - v1: 外层循环并行化（每个线程块处理一个 $Q$ 块）
   - v2: 更细粒度的并行化，减少线程块间的负载不均衡
   - 使用2D并行化：同时在 $Q$ 和 $KV$ 维度上分配工作

2. **减少非矩阵乘法操作**：
   - 优化了softmax的实现，减少了标量操作
   - 使用向量化指令处理统计量更新
   - 减少了warp级别的同步开销

3. **更好的序列并行支持**：
   - 支持跨GPU的序列维度切分
   - 优化了长序列的处理
   - 引入了因果掩码的高效处理

**性能提升的关键技术**：

**1. Warp级别的优化**：
- 利用warp shuffle指令减少共享内存访问
- 每个warp处理一个完整的行，避免跨warp通信

**2. 数据布局优化**：
- 使用转置的 $K$ 存储，提高内存合并访问
- 优化了不同头之间的数据布局

**3. 混合精度策略**：
- 累加器使用FP32保证精度
- 输入输出使用FP16/BF16节省带宽
- 动态范围调整避免数值溢出

**性能对比**（A100 GPU）：
| 序列长度 | Flash v1 | Flash v2 | 提升比例 |
|---------|----------|----------|----------|
| 512     | 1.5ms    | 0.9ms    | 1.67×    |
| 2048    | 23ms     | 12ms     | 1.92×    |
| 8192    | 370ms    | 180ms    | 2.06×    |

### 13.1.7 边缘设备上的适配考虑

在边缘设备上实现Flash Attention需要考虑：

1. **有限的SRAM大小**：
   - 移动GPU的共享内存通常只有16-32KB
   - 需要更小的块大小，可能影响效率
   - 块大小选择示例：
     - Mali GPU (16KB): $B_r = B_c = 32$
     - Adreno GPU (32KB): $B_r = B_c = 48$
     - Apple GPU (32KB): $B_r = B_c = 64$

2. **不同的内存层次**：
   - ARM架构的缓存结构与GPU不同
   - 需要针对L1/L2缓存优化块大小
   - CPU实现策略：
     - 利用L1缓存（32-64KB）作为"SRAM"
     - 块大小需要考虑缓存行（64字节）对齐
     - 使用NEON/SVE向量指令加速

3. **精度考虑**：
   - FP16/INT8量化下的数值稳定性
   - 增量softmax可能需要更高精度的累加器
   - 混合精度策略：
     - 统计量（$m_i, l_i$）使用FP32
     - 中间计算使用FP16
     - 最终输出可以是INT8

**边缘设备优化技巧**：

**1. 自适应块大小**：
```
if (seq_len < 256):
    # 短序列，使用较大块以减少开销
    block_size = min(64, seq_len)
elif (seq_len < 1024):
    # 中等序列，平衡内存和计算
    block_size = 32
else:
    # 长序列，优先考虑内存效率
    block_size = 16
```

**2. 异构计算策略**：
- 预填充阶段：使用GPU/NPU的Flash Attention
- 解码阶段：考虑CPU的标准实现（KV cache小）

**3. 量化友好的修改**：
- 使用对数域计算避免指数溢出
- 固定点数算术用于统计量更新
- 查找表（LUT）加速exp计算

**实际部署案例**：
- llama.cpp：在Apple Silicon上实现了Flash Attention变体
- MNN框架：针对移动GPU优化的分块注意力
- ONNX Runtime：支持多种边缘设备的自适应实现

## 13.2 Multi-Query/Grouped-Query Attention

### 13.2.1 多头注意力的冗余性分析

标准多头注意力（Multi-Head Attention, MHA）为每个头独立计算 $Q_h, K_h, V_h$：

$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_H)W^O$$

其中每个头：
$$\text{head}_h = \text{Attention}(QW_h^Q, KW_h^K, VW_h^V)$$

**参数和计算开销**：
- 参数量：$3Hd_{model}d_{head}$（忽略输出投影）
- KV cache大小：$2 \times \text{batch} \times H \times N \times d_{head}$

对于大模型（如 H=32），KV cache成为主要的内存瓶颈。

**冗余性观察**：
研究发现，不同注意力头的键值对存在显著相似性：
1. **相似的注意力模式**：许多头学习到类似的位置关系
2. **信息重复**：键值投影矩阵的奇异值分析显示大量冗余
3. **低秩结构**：KV空间往往可以用更低维表示

**实证分析**：
对预训练模型的分析显示：
- 相邻头之间的余弦相似度通常 > 0.8
- 前8个主成分可以解释90%以上的方差
- 许多头可以被剪枝而不显著影响性能

### 13.2.2 Multi-Query Attention (MQA)

MQA的核心思想是**所有头共享同一组键值对**：

$$\text{MQA}(Q, K, V) = \text{Concat}(\text{head}_1^{MQ}, ..., \text{head}_H^{MQ})W^O$$

其中：
$$\text{head}_h^{MQ} = \text{Attention}(Q_h, K_{shared}, V_{shared})$$

**优势**：
1. KV cache减少为原来的 $1/H$
2. 解码阶段的内存带宽需求大幅降低
3. 批量推理时效果更明显

**数学形式化**：
设 $d_k = d_{model}/H$ 为每个头的维度，则：
- MHA: $K, V \in \mathbb{R}^{N \times d_{model}}$，切分成 $H$ 份
- MQA: $K, V \in \mathbb{R}^{N \times d_k}$，所有头共享

**实现细节**：

**1. 投影矩阵设计**：
- 查询投影：$W^Q \in \mathbb{R}^{d_{model} \times d_{model}}$（保持不变）
- 键投影：$W^K \in \mathbb{R}^{d_{model} \times d_k}$（降维到单头大小）
- 值投影：$W^V \in \mathbb{R}^{d_{model} \times d_k}$（降维到单头大小）

**2. 计算流程**：
```
# 投影阶段
Q = XW^Q  # shape: [B, N, d_model]
K = XW^K  # shape: [B, N, d_k]
V = XW^V  # shape: [B, N, d_k]

# 多头切分（仅Q需要）
Q_heads = reshape(Q, [B, N, H, d_k])

# 注意力计算
for h in range(H):
    scores = Q_heads[:,:,h,:] @ K.T / sqrt(d_k)
    attn = softmax(scores)
    out_h = attn @ V
    
# 拼接输出
output = concat(all out_h) @ W^O
```

**内存节省分析**：
- 标准MHA KV cache：$2 \times B \times L \times H \times N \times d_k$
- MQA KV cache：$2 \times B \times L \times N \times d_k$
- 节省比例：$(H-1)/H$，对于H=32可节省97%

### 13.2.3 Grouped-Query Attention (GQA)

GQA是MHA和MQA的折中方案，将查询头分成 $G$ 组，每组共享KV：

$$\text{head}_h^{GQ} = \text{Attention}(Q_h, K_{g(h)}, V_{g(h)})$$

其中 $g(h) = \lfloor h \cdot G / H \rfloor$ 是头 $h$ 所属的组。

**设计空间**：
- $G = 1$：退化为MQA
- $G = H$：退化为MHA
- 典型选择：$G = H/8$ 或 $H/4$

**KV cache大小**：$2 \times \text{batch} \times G \times N \times d_{head}$

### 13.2.4 注意力变体的性能分析

**理论分析**（解码阶段）：

设批大小为 $B$，已生成长度为 $N$，则生成一个token的计算量和内存访问：

| 方法 | 计算量 (FLOPs) | KV cache读取 | 内存带宽需求 |
|------|----------------|---------------|--------------|
| MHA  | $O(BHNd_{head})$ | $O(BHNd_{head})$ | 高 |
| MQA  | $O(BHNd_{head})$ | $O(BNd_{head})$ | 低（减少$H$倍）|
| GQA  | $O(BHNd_{head})$ | $O(BGNd_{head})$ | 中等 |

**实际性能考虑**：
1. **计算/访存比**：
   - MQA提高了计算密度，更适合memory-bound场景
   - 在边缘设备上效果尤其明显

2. **并行度**：
   - MHA可以完全并行计算所有头
   - MQA/GQA需要广播共享的KV，可能影响并行效率

### 13.2.5 从MHA到MQA/GQA的转换

**训练时转换**：
1. **初始化策略**：
   - 平均池化：$K_{shared} = \frac{1}{H}\sum_{h=1}^H K_h$
   - 选择特定头：$K_{shared} = K_1$（通常选择第一个头）

2. **渐进式转换**：
   ```
   α = min(1, training_step / warmup_steps)
   K_effective = α * K_shared + (1-α) * K_original
   ```

**推理时近似**（无需重训练）：
1. **平均池化方法**：
   $$K_{MQA} = \frac{1}{H}\sum_{h=1}^H K_h^{MHA}, \quad V_{MQA} = \frac{1}{H}\sum_{h=1}^H V_h^{MHA}$$

2. **主成分分析（PCA）**：
   - 对 $[K_1, ..., K_H]$ 进行SVD分解
   - 取最大奇异值对应的向量作为共享KV

### 13.2.6 边缘部署的实践考虑

1. **内存布局优化**：
   - MQA/GQA的KV需要高效的广播机制
   - 考虑SIMD指令的对齐要求

2. **量化策略**：
   - 共享KV可能需要更高的量化精度
   - Per-head量化 vs Per-tensor量化的权衡

3. **动态切换**：
   - 根据序列长度动态选择MHA/GQA
   - 短序列用MHA保持质量，长序列用GQA节省内存

### 13.2.7 实验结果与分析

以LLaMA系列模型为例：

| 模型 | 原始(MHA) | GQA-8 | GQA-4 | MQA |
|------|-----------|--------|--------|-----|
| PPL提升 | 0% | +0.2% | +0.5% | +1.2% |
| KV cache | 100% | 12.5% | 25% | 3.1% |
| 解码速度提升 | 1x | 1.8x | 1.5x | 2.2x |

关键发现：
1. GQA-8（8组）在质量损失很小的情况下获得显著加速
2. MQA在极长序列（>4K）时优势最明显
3. 不同任务对共享KV的敏感度不同，知识密集型任务损失较大

## 13.3 稀疏注意力模式

### 13.3.1 注意力稀疏性的动机

完整注意力的 $O(N^2)$ 复杂度在长序列上变得不可承受。然而，实证研究表明：

1. **注意力分布的长尾特性**：大部分注意力权重接近于0
2. **局部性**：相邻token之间的注意力通常更强
3. **全局锚点**：某些特殊token（如[CLS]）需要全局信息

这些观察启发了各种稀疏注意力模式的设计。

### 13.3.2 固定稀疏模式

**1. 窗口注意力（Window Attention）**

每个token只关注固定窗口内的其他token：

$$S_{ij} = \begin{cases}
\frac{Q_iK_j^T}{\sqrt{d_k}} & \text{if } |i-j| \leq w \\
-\infty & \text{otherwise}
\end{cases}$$

其中 $w$ 是窗口半径。

**复杂度**：$O(Nw)$ 而非 $O(N^2)$

**2. 跨步注意力（Strided Attention）**

固定步长的稀疏连接：

$$\text{mask}(i,j) = \mathbb{1}[(i-j) \bmod s = 0]$$

**3. 组合模式（Combination Patterns）**

Longformer提出的组合模式：
- 滑动窗口注意力（局部）
- 扩张窗口注意力（中等距离）
- 全局注意力（特定token）

数学表示：
$$\text{Attention}_i = \begin{cases}
\text{WindowAttn}_i & \text{if } i \in \text{LocalTokens} \\
\text{GlobalAttn}_i & \text{if } i \in \text{GlobalTokens}
\end{cases}$$

### 13.3.3 学习型稀疏模式

**1. 基于阈值的动态稀疏**

计算完整注意力分数后，保留top-k或超过阈值的连接：

$$P_{ij} = \begin{cases}
\text{softmax}(S_{ij}) & \text{if } S_{ij} \in \text{top-k}(S_i) \\
0 & \text{otherwise}
\end{cases}$$

**问题**：需要先计算完整的 $S = QK^T$，无法节省计算

**2. 可学习的稀疏掩码**

引入可学习的二值掩码 $M \in \{0,1\}^{N \times N}$：

$$\text{SparseAttn}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} \odot M\right)V$$

使用Gumbel-Softmax或直通估计器（STE）进行训练。

### 13.3.4 分层稀疏注意力

**BigBird的设计**：结合三种注意力模式

1. **随机注意力**：每个token随机关注 $r$ 个其他token
2. **窗口注意力**：关注邻近的 $w$ 个token  
3. **全局注意力**：$g$ 个全局token与所有其他token相连

总的连接数：$O(N(r + w + g))$

**数学形式化**：
设注意力图 $G = (V, E)$，其中：
- $E_{window} = \{(i,j) : |i-j| \leq w\}$
- $E_{random} = \{(i,j) : (i,j) \in \text{RandomSample}(r)\}$
- $E_{global} = \{(i,j) : i \in G \text{ or } j \in G\}$

则 $E = E_{window} \cup E_{random} \cup E_{global}$

### 13.3.5 稀疏注意力的高效实现

**1. 块稀疏格式（Block Sparse Format）**

将注意力矩阵分成 $B \times B$ 的块，只计算非零块：

```
稀疏掩码（块级别）：
[1 1 0 0]
[1 1 1 0]  
[0 1 1 1]
[0 0 1 1]
```

**优势**：
- 更好的硬件利用率（块级别的矩阵乘法）
- 减少索引开销

**2. CSR格式存储**

对于极度稀疏的模式，使用压缩稀疏行（CSR）格式：
- values: 非零元素值
- col_indices: 列索引
- row_ptr: 每行的起始位置

**3. 核融合优化**

避免生成完整的注意力矩阵：
```
for each query i:
    sparse_indices = get_sparse_pattern(i)
    K_sparse = gather(K, sparse_indices)
    S_sparse = Q[i] @ K_sparse.T
    P_sparse = softmax(S_sparse)
    O[i] = P_sparse @ gather(V, sparse_indices)
```

### 13.3.6 稀疏模式的选择策略

**1. 基于任务的选择**：
- 文档理解：需要长距离依赖，适合BigBird模式
- 对话生成：局部连贯性重要，窗口注意力足够
- 代码生成：需要结构化稀疏（如语法树引导）

**2. 基于硬件的选择**：
- GPU：块稀疏模式，利用tensor core
- CPU：CSR格式，优化缓存访问
- DSP：固定模式，便于向量化

**3. 动态选择**：
根据序列长度动态切换：
```
if seq_len < 512:
    use_full_attention()
elif seq_len < 2048:
    use_window_attention(window=256)
else:
    use_bigbird_attention()
```

### 13.3.7 稀疏注意力的理论保证

**表达能力分析**：

定理：对于窗口大小 $w = O(\log N)$，$L$ 层的窗口注意力可以模拟完整注意力。

证明要点：
- 每层扩展感受野 $2w$
- $L$ 层后的感受野：$2Lw = O(L\log N)$
- 当 $L = O(N/\log N)$ 时可覆盖整个序列

**近似误差界**：

对于top-k稀疏：
$$\|P_{full} - P_{sparse}\|_F \leq \epsilon$$

其中 $k = O(N\log N/\epsilon^2)$ 即可保证 $\epsilon$ 误差。

## 13.4 线性注意力机制

### 13.4.1 线性注意力的核心思想

标准注意力的计算瓶颈在于 $\text{softmax}(QK^T)$ 的矩阵乘法。线性注意力通过**分解注意力矩阵**来避免显式计算 $N \times N$ 的矩阵。

**核心变换**：
将 $\text{softmax}(QK^T)$ 近似为 $\phi(Q)\phi(K)^T$，其中 $\phi$ 是特征映射。

这样可以改变计算顺序：
$$\text{Attention}(Q,K,V) = \phi(Q)[\phi(K)^TV]$$

计算顺序的改变将复杂度从 $O(N^2d)$ 降至 $O(Nd^2)$。

### 13.4.2 核方法视角

**Softmax作为核函数**：

标准注意力可以写成：
$$A_{ij} = \frac{\exp(Q_iK_j^T/\sqrt{d})}{\sum_k \exp(Q_iK_k^T/\sqrt{d})} = \frac{k(Q_i, K_j)}{\sum_k k(Q_i, K_k)}$$

其中 $k(x,y) = \exp(x^Ty/\sqrt{d})$ 是指数核。

**核函数的分解**：

如果存在特征映射 $\phi$ 使得：
$$k(x,y) = \langle\phi(x), \phi(y)\rangle$$

则可以实现线性复杂度的注意力。

### 13.4.3 具体的线性注意力方法

**1. Linear Transformer (Katharopoulos et al.)**

使用简单的特征映射：
$$\phi(x) = \text{elu}(x) + 1$$

其中elu是指数线性单元。这保证了 $\phi(x) \geq 0$。

**因果掩码的处理**：
对于自回归生成，需要因果掩码。线性注意力通过RNN形式实现：

$$S_i = S_{i-1} + \phi(K_i)V_i^T$$
$$O_i = \frac{\phi(Q_i)S_i}{\phi(Q_i)\sum_{j \leq i}\phi(K_j)}$$

**2. Performer (Choromanski et al.)**

使用随机特征近似softmax核：

$$\phi(x) = \frac{\exp(\|x\|^2/2)}{\sqrt{m}} [\exp(w_1^Tx), ..., \exp(w_m^Tx)]^T$$

其中 $w_i \sim \mathcal{N}(0, I)$ 是随机投影向量。

**理论保证**：
当 $m = O(d\log d/\epsilon^2)$ 时，近似误差小于 $\epsilon$。

**3. RFA (Random Feature Attention)**

使用确定性的正交随机特征：
1. 生成正交矩阵 $W \in \mathbb{R}^{d \times m}$
2. $\phi(x) = [\sin(Wx), \cos(Wx)]^T / \sqrt{m}$

优势：更稳定的近似，更少的随机性。

### 13.4.4 线性注意力的统一框架

**一般形式**：
$$\text{LinearAttn}(Q,K,V) = \frac{\phi(Q)[\phi(K)^TV]}{\phi(Q)[\phi(K)^T\mathbf{1}]}$$

其中分母项用于归一化。

**设计空间**：
1. **特征映射选择**：
   - 恒等映射：$\phi(x) = x$（需要非负约束）
   - ReLU族：$\phi(x) = \text{ReLU}(x)$
   - 指数族：$\phi(x) = \exp(x/\tau)$
   - 随机特征：如Performer

2. **归一化策略**：
   - L2归一化：$\phi(x) = x/\|x\|_2$
   - Softmax归一化：在特征维度上
   - 无归一化：依赖训练时的正则化

### 13.4.5 线性注意力的优化技巧

**1. 数值稳定性**

问题：当 $\phi(K)^T\mathbf{1}$ 接近0时，除法不稳定。

解决方案：
- 添加小常数：$\phi(Q)[\phi(K)^T\mathbf{1}] + \epsilon$
- 使用对数空间计算
- 梯度裁剪

**2. 特征维度选择**

权衡：
- 更高维度 $m$：更好的近似，更多计算
- 建议：$m = O(\sqrt{N})$ 时达到计算-精度平衡

**3. 混合精度策略**

- 特征映射用FP32计算（数值稳定性）
- 矩阵乘法用FP16/BF16（速度）
- 累加器用FP32（精度）

### 13.4.6 线性注意力的应用场景

**1. 超长序列处理**

当 $N \gg d$ 时，线性注意力优势明显：
- 标准注意力：$O(N^2d)$
- 线性注意力：$O(Nd^2)$

临界点：$N = d$ 时两者计算量相当。

**2. 流式/在线推理**

RNN形式的线性注意力支持：
- 常数内存的增量计算
- 无需存储完整的KV cache
- 适合边缘设备的实时应用

**3. 跨模态注意力**

图像-文本等跨模态场景，序列长度差异大：
- 图像：$N_{img} = 14 \times 14 = 196$（ViT）
- 文本：$N_{text} = 512$
- 交叉注意力：$O(N_{img} \times N_{text})$ → $O((N_{img} + N_{text})d)$

### 13.4.7 实验结果与分析

**在不同任务上的表现**：

| 方法 | 语言建模 (PPL) | 图像分类 (Acc) | 长文本QA (F1) |
|------|----------------|----------------|----------------|
| 标准注意力 | 15.2 | 81.5% | 73.4 |
| Performer | 16.1 (+6%) | 80.8% | 71.2 |
| Linear Transformer | 16.8 (+10%) | 79.6% | 69.8 |
| 混合方案* | 15.5 (+2%) | 81.2% | 72.9 |

*混合方案：前几层用标准注意力，后续层用线性注意力

**关键发现**：
1. 线性注意力在局部依赖任务上表现良好
2. 全局推理任务（如算术）性能下降明显
3. 混合架构能够较好平衡效率和性能

## 本章小结

本章系统地探讨了注意力机制的各种优化技术，这些技术对于在资源受限的边缘设备上部署大语言模型至关重要：

1. **Flash Attention**通过平铺和重计算策略，将内存访问从 $O(N^2)$ 降至 $O(N)$，在保持精确计算的同时大幅提升了硬件利用率。其核心在于利用GPU的内存层次结构，通过分块计算和增量softmax避免了中间结果的频繁读写。

2. **Multi-Query和Grouped-Query Attention**通过在多个查询头之间共享键值对，将KV cache的大小降低了数倍到数十倍。这种方法特别适合解码阶段和长序列场景，在质量损失很小的情况下获得了显著的加速。

3. **稀疏注意力模式**利用了注意力分布的稀疏性，通过固定模式（窗口、跨步）或学习型模式将计算复杂度从 $O(N^2)$ 降至 $O(N\log N)$ 或 $O(N)$。BigBird等方法通过组合局部、随机和全局注意力，在保持模型表达能力的同时实现了高效计算。

4. **线性注意力机制**通过核方法和特征映射，将注意力计算的复杂度降至 $O(Nd^2)$。虽然在某些任务上有性能损失，但其常数内存的特性使其特别适合流式处理和超长序列。

**关键公式回顾**：

- Flash Attention的增量softmax更新：
  $$O_i^{(j)} = \frac{e^{m_i^{(j-1)} - m_i^{(j)}} l_i^{(j-1)}}{l_i^{(j)}} O_i^{(j-1)} + \frac{1}{l_i^{(j)}} \sum_k e^{S_{ijk} - m_i^{(j)}} V_{jk}$$

- GQA的头分组映射：
  $$g(h) = \lfloor h \cdot G / H \rfloor$$

- 线性注意力的核心变换：
  $$\text{Attention}(Q,K,V) = \phi(Q)[\phi(K)^TV]$$

**实践建议**：

1. 对于边缘部署，优先考虑GQA（特别是8组配置），它提供了最佳的质量-效率平衡
2. Flash Attention在有足够SRAM的设备上效果最好，需要根据硬件调整块大小
3. 稀疏注意力适合特定的应用场景，需要根据任务特性选择合适的稀疏模式
4. 线性注意力可以作为混合架构的一部分，在模型的高层使用以处理长距离依赖

## 练习题

### 基础题

1. **Flash Attention的内存访问分析**
   计算标准注意力和Flash Attention在序列长度N=2048、特征维度d=64、块大小B=64时的HBM访问次数。假设使用FP16存储。
   
   *Hint*：考虑每个矩阵元素的读写次数，以及中间结果的存储。

2. **MQA的KV Cache计算**
   对于一个32头的模型，批大小B=8，序列长度N=1024，每个头维度为128，计算MHA、GQA-8和MQA的KV cache内存占用（以MB为单位）。
   
   *Hint*：KV cache = 2 × batch × heads × seq_len × head_dim × bytes_per_element

3. **稀疏注意力的连接数**
   对于BigBird注意力，如果窗口大小w=3，随机连接数r=2，全局token数g=2，序列长度N=512，计算总的注意力连接数和稀疏度。
   
   *Hint*：稀疏度 = 1 - (实际连接数 / N²)

### 挑战题

4. **Flash Attention的最优块大小**
   给定SRAM大小为48KB，需要存储Q块、K块、V块以及中间结果。在FP16精度下，推导使SRAM利用率最大化的块大小公式。考虑需要额外存储每行的最大值和求和结果。
   
   *Hint*：设块大小为B_r × B_c，列出所有需要存储的张量及其大小。

5. **线性注意力的误差界分析**
   证明：对于Performer使用m个随机特征时，注意力矩阵的近似误差期望满足：
   $$\mathbb{E}[\|\hat{A} - A\|_F] \leq \frac{C}{\sqrt{m}} \|A\|_F$$
   其中C是与维度相关的常数。
   
   *Hint*：使用随机特征的方差分析和矩阵范数的性质。

6. **混合注意力架构设计**
   设计一个12层Transformer的注意力配置，要求：
   - 前4层保持完整注意力以捕获局部模式
   - 中间4层使用GQA-4以平衡效率
   - 最后4层使用窗口注意力（窗口256）+ 线性注意力的混合
   
   分析这种设计在不同序列长度（512, 2048, 8192）下相对于全MHA的计算节省和内存节省。
   
   *Hint*：分别计算每种配置的FLOPs和内存占用，考虑批大小的影响。

7. **稀疏模式的表达能力**
   考虑一个只使用窗口大小为w的局部注意力的L层Transformer。如果要保证任意两个位置的信息能够交互，w和L需要满足什么关系？对于序列长度N=1024，如果限制w≤32，最少需要多少层？
   
   *Hint*：考虑信息传播的"感受野"概念。

8. **注意力优化的能效分析**
   假设一个边缘设备的内存带宽为25.6 GB/s，计算性能为2 TFLOPS（FP16）。对于批大小1、序列长度512、模型维度768的注意力计算，分析标准注意力、Flash Attention和GQA-8分别是compute-bound还是memory-bound。计算各自的硬件利用率。
   
   *Hint*：计算arithmetic intensity（FLOPs/字节），与硬件的计算/带宽比值对比。

### 答案

<details>
<summary>点击查看答案</summary>

1. **标准注意力**：
   - 读Q,K,V: 3×N×d×2 = 3×2048×64×2 = 786KB
   - 写S=QK^T: N²×2 = 8MB
   - 读S计算softmax: 8MB
   - 写P: 8MB
   - 读P,V计算输出: 8MB + 256KB
   - 写输出: 256KB
   - 总计：约32.5MB
   
   **Flash Attention**：
   - 读Q,K,V各一次: 786KB
   - 写输出: 256KB
   - 总计：约1MB

2. MHA: 2×8×32×1024×128×2 = 128MB
   GQA-8: 2×8×8×1024×128×2 = 32MB
   MQA: 2×8×1×1024×128×2 = 4MB

3. 每个token的连接数：
   - 窗口: 2w+1 = 7
   - 随机: r = 2
   - 全局: 2g = 4（双向）
   - 总计: 13连接/token（考虑重复）
   - 总连接数: ≈512×13 = 6656
   - 稀疏度: 1 - 6656/(512²) ≈ 97.5%

4. 需要存储：
   - Q块: B_r × d × 2字节
   - K,V块: 2 × B_c × d × 2字节
   - S块: B_r × B_c × 2字节
   - 统计量: B_r × 8字节（max和sum各用FP32）
   
   约束：2B_r×d + 4B_c×d + 2B_r×B_c + 8B_r ≤ 48KB
   
   当B_r = B_c = B时，简化为：6Bd + 2B² + 8B ≤ 48KB
   对于d=64，最优B ≈ 48

5. 证明思路：
   - Performer的特征映射引入的误差来自随机投影
   - 每个随机特征的方差为O(1/m)
   - 使用矩阵集中不等式和Frobenius范数的次可加性
   - 最终得到期望误差界O(1/√m)

6. 计算节省（以N=2048为例）：
   - 层1-4：标准MHA，100%计算
   - 层5-8：GQA-4，KV cache减少75%，解码加速约1.6x
   - 层9-12：窗口(256) + 线性混合，计算复杂度O(N×256 + N×d²)
   - 总体计算节省：约45%
   - 内存节省：约60%（主要来自KV cache）

7. 每层扩展感受野2w，L层后感受野为2Lw
   需要：2Lw ≥ N-1
   因此：L ≥ (N-1)/(2w)
   
   对于N=1024, w=32：L ≥ 1023/64 ≈ 16层

8. **标准注意力**：
   - 计算量：4×N²×d = 4×512²×768 ≈ 0.8 GFLOPS
   - 内存访问：约20MB
   - Arithmetic Intensity：0.8G/20M ≈ 40 FLOPs/byte
   - 硬件AI：2T/25.6G ≈ 78 FLOPs/byte
   - 结论：Memory-bound，利用率约51%
   
   **Flash Attention**：
   - 内存访问减少到约2MB
   - AI提升到约400 FLOPs/byte
   - 结论：Compute-bound，利用率约40%
   
   **GQA-8**：
   - KV读取减少8倍
   - 解码阶段更接近compute-bound
   - 利用率提升到约70%

</details>
