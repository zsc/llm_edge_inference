# 第9章：模型剪枝

模型剪枝是一种通过移除神经网络中冗余或不重要的参数来压缩模型的技术。在边缘设备部署场景中，剪枝不仅能够减少模型的存储需求，还能显著降低推理时的计算量和内存占用。本章将深入探讨剪枝的数学原理、实践策略以及在大语言模型中的应用，重点关注如何在保持模型性能的同时最大化压缩率。

## 9.1 结构化剪枝vs非结构化剪枝

### 9.1.1 非结构化剪枝

非结构化剪枝（Unstructured Pruning）是指在权重矩阵中任意位置移除单个权重元素，不考虑特定的结构模式。这种方法最早可以追溯到1990年代的Optimal Brain Damage (OBD)和Optimal Brain Surgeon (OBS)工作。

**数学表示**：
对于权重矩阵 $\mathbf{W} \in \mathbb{R}^{m \times n}$，非结构化剪枝通过二值掩码矩阵 $\mathbf{M} \in \{0,1\}^{m \times n}$ 实现：

$$\mathbf{W}_{\text{pruned}} = \mathbf{W} \odot \mathbf{M}$$

其中 $\odot$ 表示逐元素乘法（Hadamard积）。

**稀疏度定义**：
稀疏度 $s$ 定义为零元素的比例：

$$s = \frac{\|\mathbf{M}\|_0}{mn} = \frac{\text{非零元素数量}}{\text{总元素数量}}$$

**历史发展轨迹**：
1. **Optimal Brain Damage (OBD, 1990)**：LeCun等人首次提出基于Hessian对角近似的剪枝方法
2. **Optimal Brain Surgeon (OBS, 1993)**：Hassibi和Stork扩展到完整Hessian矩阵
3. **Magnitude Pruning (2015)**：Han等人证明简单的幅度剪枝在深度网络中同样有效
4. **Lottery Ticket Hypothesis (2018)**：Frankle和Carbin发现稀疏子网络的存在性

**稀疏存储格式**：
非结构化稀疏矩阵通常使用以下格式存储：
- **COO (Coordinate)格式**：存储 (row, col, value) 三元组
- **CSR (Compressed Sparse Row)格式**：使用行指针数组、列索引数组和值数组
- **CSC (Compressed Sparse Column)格式**：类似CSR但按列存储

对于稀疏度为 $s$ 的矩阵，COO格式的存储需求从 $mn$ 降至 $3(1-s)mn$，只有当 $s > 2/3$ 时才有存储优势。

**存储格式详细分析**：

1. **COO格式实例**：
   对于4×4矩阵，70%稀疏度：
   ```
   原始矩阵：          COO表示：
   [0  0  3  0]       row: [0, 1, 2, 3]
   [0  5  0  0]       col: [2, 1, 0, 2]  
   [1  0  0  0]       val: [3, 5, 1, 7]
   [0  0  7  0]
   ```
   存储需求：12个值（3×4）vs 16个原始值

2. **CSR格式优势**：
   - 行指针：$O(m)$ 存储
   - 列索引：$O(\text{nnz})$ 存储  
   - 数值：$O(\text{nnz})$ 存储
   - 总存储：$O(m + 2\text{nnz})$

3. **块稀疏存储（BCSR）**：
   将矩阵划分为 $b \times b$ 块，只存储非零块：
   $$\text{Storage} = \text{block\_indices} + b^2 \times \text{num\_blocks}$$
   
   当块内稠密度高时更有效。

**优势**：
- 理论上可以达到更高的压缩率（可达99%以上稀疏度）
- 灵活性高，可以精确控制剪枝粒度
- 保持模型精度的能力更强
- 对于极高稀疏度（>95%），可以显著减少存储

**劣势**：
- 需要专门的稀疏矩阵运算库支持（如cuSPARSE）
- 在通用硬件（CPU/GPU）上难以获得实际加速
- 存储需要额外的索引信息（通常需要16-32位整数）
- 内存访问模式不规则，缓存利用率低
- 负载不均衡问题难以解决

**实际加速分析**：
稀疏矩阵乘法的实际性能受多个因素影响：
$$T_{\text{sparse}} = T_{\text{compute}} + T_{\text{memory}} + T_{\text{overhead}}$$

其中：
- $T_{\text{compute}} \propto (1-s)mn$：计算时间与非零元素成正比
- $T_{\text{memory}}$：受限于不规则内存访问模式
- $T_{\text{overhead}}$：包括索引解码、负载均衡等开销

通常只有当稀疏度超过90-95%时，在GPU上才能获得实际加速。

**稀疏矩阵运算优化技术**：

1. **重排序优化**：
   通过矩阵行列重排序减少缓存缺失：
   - Cuthill-McKee算法：最小化带宽
   - Nested Dissection：递归分割图
   - AMD (Approximate Minimum Degree)：最小化填充

2. **向量化策略**：
   - **ELL格式**：固定每行非零元素数，适合GPU
   - **Sliced ELL**：分片处理不同稠密度区域
   - **混合格式**：ELL+COO处理长尾分布

3. **负载均衡技术**：
   ```
   动态分配策略：
   - 行合并：将短行合并处理
   - 动态调度：根据非零元素数分配线程
   - 两阶段处理：先处理规则部分，再处理不规则部分
   ```

**边缘设备上的非结构化剪枝**：

在资源受限的边缘设备上，非结构化剪枝面临特殊挑战：

1. **内存带宽限制**：
   边缘设备的内存带宽通常是瓶颈：
   $$\text{Bandwidth}_{\text{required}} = \frac{\text{nnz} \times (\text{sizeof(value)} + \text{sizeof(index)})}{\text{time}}$$
   
   对于移动GPU（如Mali G78），带宽约为50GB/s，远低于桌面GPU。

2. **缓存层次利用**：
   - L1缓存：32-64KB，需要精心的数据布局
   - L2缓存：256KB-2MB，关键的性能决定因素
   - 系统内存：4-8GB，访问延迟高

3. **能效考虑**：
   稀疏运算的能效分析：
   $$\text{Energy} = E_{\text{compute}} + E_{\text{memory}} + E_{\text{control}}$$
   
   其中控制开销 $E_{\text{control}}$ 在稀疏运算中占比可达30-40%。

### 9.1.2 结构化剪枝

结构化剪枝（Structured Pruning）按照预定义的结构模式移除参数，如整个通道、滤波器或注意力头。这种方法的核心优势在于剪枝后的模型仍然保持规则的张量结构，可以直接利用现有的高度优化的密集计算库。

**主要类型**：

1. **通道剪枝（Channel Pruning）**：
   对于卷积层权重 $\mathbf{W} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K \times K}$，通道剪枝移除整个输入或输出通道：
   
   $$\mathbf{W}_{\text{pruned}} \in \mathbb{R}^{C'_{\text{out}} \times C'_{\text{in}} \times K \times K}$$
   
   其中 $C'_{\text{out}} \leq C_{\text{out}}$，$C'_{\text{in}} \leq C_{\text{in}}$
   
   计算量减少比例：
   $$\text{FLOPs}_{\text{reduction}} = 1 - \frac{C'_{\text{out}} \times C'_{\text{in}}}{C_{\text{out}} \times C_{\text{in}}}$$

2. **滤波器剪枝（Filter Pruning）**：
   移除整个滤波器（输出通道），这是最常用的结构化剪枝形式：
   
   $$\mathbf{W}_{i,:,:,:} = \mathbf{0}, \quad i \in \mathcal{P}$$
   
   其中 $\mathcal{P}$ 是被剪枝的滤波器索引集合。
   
   **级联效应**：剪枝第 $l$ 层的输出通道会影响第 $l+1$ 层的输入通道：
   $$C_{\text{in}}^{(l+1)} = C_{\text{out}}^{(l)} - |\mathcal{P}^{(l)}|$$

3. **注意力头剪枝（Attention Head Pruning）**：
   对于多头注意力，移除整个注意力头：
   
   $$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_i)_{i \notin \mathcal{P}} \mathbf{W}^O$$
   
   每个头的计算复杂度为 $O(n^2 d_h)$，其中 $n$ 是序列长度，$d_h = d_{\text{model}}/h$ 是每个头的维度。

4. **块剪枝（Block Pruning）**：
   将权重矩阵划分为 $b \times b$ 的块，以块为单位进行剪枝：
   
   $$\mathbf{W} = \begin{bmatrix}
   \mathbf{B}_{11} & \mathbf{B}_{12} & \cdots \\
   \mathbf{B}_{21} & \mathbf{B}_{22} & \cdots \\
   \vdots & \vdots & \ddots
   \end{bmatrix}$$
   
   其中 $\mathbf{B}_{ij} \in \mathbb{R}^{b \times b}$ 要么全零要么保留。

**硬件友好性分析**：

结构化剪枝后的计算可以直接映射到密集矩阵运算：
- 无需特殊的稀疏运算支持
- 可以直接使用现有的BLAS/cuBLAS库
- 内存访问模式规则，缓存友好
- 向量化指令（如AVX-512）利用率高

**实际加速效果**：
对于结构化剪枝，实际加速比接近理论值：
$$\text{Speedup}_{\text{actual}} \approx \frac{\text{FLOPs}_{\text{original}}}{\text{FLOPs}_{\text{pruned}}} \times \eta$$

其中效率因子 $\eta \in [0.8, 0.95]$，取决于硬件架构和实现质量。

**与批归一化的交互**：
通道剪枝需要特别处理批归一化层：
- 剪枝的通道对应的BN参数（$\gamma$, $\beta$, running_mean, running_var）也需要移除
- 可以利用BN的缩放因子 $\gamma$ 作为通道重要性的指标

### 9.1.3 半结构化剪枝

近年来，NVIDIA等硬件厂商开始支持半结构化稀疏模式，如2:4稀疏（每4个连续元素中保留2个）。这种方法试图在非结构化剪枝的灵活性和结构化剪枝的硬件效率之间找到平衡点。

**2:4稀疏模式**：
对于向量 $\mathbf{v} = [v_1, v_2, v_3, v_4]$，2:4稀疏要求恰好两个元素为零：

$$\|\mathbf{v}\|_0 = 2$$

这种模式在NVIDIA Ampere架构（A100）及更新的GPU上通过Sparse Tensor Core可以获得理论上2倍的加速。

**硬件实现原理**：
NVIDIA的Sparse Tensor Core通过专门的硬件单元实现2:4稀疏加速：
1. **压缩存储**：每4个元素只存储2个非零值及其2-bit索引
2. **并行计算**：硬件自动处理稀疏模式的解码和计算
3. **流水线优化**：稀疏解码与计算重叠执行

**数学优化问题**：
寻找最优的2:4稀疏模式可以表述为：

$$\min_{\mathbf{M}} \|\mathbf{W} - \mathbf{W} \odot \mathbf{M}\|_F^2$$
$$\text{s.t. } \sum_{j=4k}^{4k+3} M_{ij} = 2, \quad \forall i,k$$

这是一个组合优化问题，对于每组4个元素，需要从 $\binom{4}{2} = 6$ 种可能中选择最优方案。

**贪婪算法**：
对每组4个连续元素，保留绝对值最大的2个：
```
For each group [w1, w2, w3, w4]:
    indices = argsort(|wi|, descending)
    keep indices[0] and indices[1]
    prune indices[2] and indices[3]
```

**SR-STE (Sparse-Refined Straight-Through Estimator)**：
NVIDIA提出的训练时方法，使用可微分的top-k操作：

$$\mathbf{W}_{\text{sparse}} = \text{SR-STE}(\mathbf{W}) = \mathbf{W} \odot \text{TopK-Mask}(\mathbf{W}, k=2, \text{group}=4)$$

反向传播时使用直通估计器，允许被剪枝权重的梯度流动。

**其他半结构化模式**：
1. **N:M稀疏**：每M个元素中保留N个
   - 1:4 稀疏：75%稀疏度，4倍理论加速
   - 2:8 稀疏：75%稀疏度，更大的选择空间
   - 4:8 稀疏：50%稀疏度，精度损失更小

2. **向量稀疏（Vector-wise Sparsity）**：
   整个向量要么全保留要么全剪枝，适合SIMD架构：
   $$\mathbf{W} = [\mathbf{v}_1, \mathbf{v}_2, ..., \mathbf{v}_n], \quad \mathbf{v}_i \in \{0\}^d \cup \mathbb{R}^d$$

3. **图案稀疏（Pattern-based Sparsity）**：
   预定义的稀疏模式，如棋盘格、条纹等：
   $$\mathbf{M}_{ij} = f(i, j)$$
   其中 $f$ 是预定义的模式函数。

**硬件支持现状**：
- **NVIDIA A100/H100**: 支持2:4稀疏，通过Sparse Tensor Core
- **AMD MI300**: 支持类似的结构化稀疏
- **Intel Sapphire Rapids**: 支持向量稀疏加速
- **Apple Neural Engine**: 支持块稀疏模式

**性能分析**：
2:4稀疏的实际加速取决于多个因素：
$$\text{Speedup}_{\text{2:4}} = \frac{2}{1 + \alpha}$$

其中 $\alpha$ 是额外开销比例，包括：
- 稀疏索引编码/解码
- 内存带宽限制
- 负载不均衡

实践中，$\alpha \approx 0.2-0.5$，因此实际加速比约为1.3-1.7倍。

### 9.1.4 剪枝模式选择策略

选择合适的剪枝模式需要考虑多个因素，这是一个多目标优化问题，需要在模型大小、推理速度、精度和硬件兼容性之间权衡。

**决策框架**：
剪枝模式选择可以形式化为多目标优化问题：
$$\min_{\mathbf{M}} \{\mathcal{L}_{\text{accuracy}}(\mathbf{M}), \mathcal{L}_{\text{latency}}(\mathbf{M}), \mathcal{L}_{\text{memory}}(\mathbf{M}), \mathcal{L}_{\text{energy}}(\mathbf{M})\}$$

使用Pareto优化找到非支配解集。

1. **硬件特性与剪枝模式匹配**：
   - **通用CPU/GPU**：优先选择结构化剪枝
     - CPU：通道剪枝最优，可利用SIMD指令
     - GPU：滤波器剪枝，保持warp执行效率
   - **专用加速器**：
     - NVIDIA A100+：2:4半结构化
     - TPU：块剪枝（128×128块）
     - Qualcomm Hexagon DSP：向量稀疏
   - **移动端NPU**：需要根据具体架构
     - Apple Neural Engine：支持通道剪枝
     - 高通Hexagon：支持滤波器和通道剪枝

**硬件能力评估矩阵**：
```
硬件类型         | 非结构化 | 结构化 | 2:4稀疏 | 块稀疏
----------------|---------|--------|---------|--------
CPU (x86/ARM)   |    ✗    |   ✓✓   |    ✗    |   ✓
GPU (消费级)     |    ✗    |   ✓✓   |    ✗    |   ✓
GPU (A100/H100) |    ✓    |   ✓✓   |   ✓✓    |   ✓
移动GPU         |    ✗    |   ✓✓   |    ✗    |   ✓
DSP/NPU         |    ✗    |   ✓    |    ✗    |   ✓

✓✓: 硬件优化支持  ✓: 可运行  ✗: 无加速或不支持
```

2. **压缩率要求与方法选择**：
   - **极高压缩率（>95%稀疏）**：
     - 首选：非结构化剪枝 + 量化组合
     - 备选：级联结构化剪枝
   - **高压缩率（90-95%）**：
     - 首选：混合剪枝（结构化+非结构化）
     - 备选：aggressive结构化剪枝
   - **中等压缩率（50-90%）**：
     - 首选：结构化剪枝
     - 备选：2:4半结构化（如果硬件支持）
   - **低压缩率（<50%）**：
     - 首选：通道剪枝
     - 备选：注意力头剪枝

3. **模型组件的剪枝敏感性分析**：
   
   对于Transformer/LLM模型，经验性的剪枝敏感性排序：
   
   $$\text{Embedding} < \text{FFN}_{\text{up}} < \text{FFN}_{\text{down}} < \text{Attention}_{\text{V}} < \text{Attention}_{\text{K,Q}} < \text{LayerNorm}$$
   
   **详细分析**：
   - **Embedding层**：可承受高达70%的剪枝，通过维度缩减
   - **FFN上投影**（FFN_up）：可剪枝50-60%，hidden_dim从4d减至2d
   - **FFN下投影**（FFN_down）：可剪枝40-50%
   - **注意力V投影**：可剪枝30-40%
   - **注意力K,Q投影**：敏感，建议<30%剪枝
   - **LayerNorm**：极其敏感，不建议剪枝

4. **层级剪枝率分配策略**：
   
   基于泰勒展开的敏感度分配：
   $$p_l = p_{\text{avg}} \cdot \left(1 + \lambda \cdot \frac{S_{\text{avg}} - S_l}{S_{\text{avg}}}\right)$$
   
   其中：
   - $p_l$：第$l$层的剪枝率
   - $p_{\text{avg}}$：平均目标剪枝率
   - $S_l$：第$l$层的敏感度得分
   - $\lambda$：调节因子，通常为0.5

5. **混合剪枝策略**：
   
   现代实践中，常采用混合策略以获得最佳效果：
   
   ```
   Layer Type          | Pruning Method      | Typical Rate
   -------------------|--------------------|--------------
   Embedding          | Structured (dim)    | 50-70%
   Attention Q,K,V    | Structured (head)   | 25-40%
   Attention Output   | Semi-structured     | 40-50%
   FFN Layer 1        | Structured (neuron) | 50-60%
   FFN Layer 2        | Semi-structured     | 40-50%
   Final Linear       | Unstructured        | 70-80%
   ```

6. **动态剪枝模式选择**：
   
   根据运行时条件动态选择剪枝模式：
   - **高吞吐场景**：使用aggressive结构化剪枝
   - **高精度场景**：使用conservative非结构化剪枝
   - **功耗受限场景**：优先剪枝计算密集层

7. **剪枝与其他优化技术的协同**：
   - **剪枝+量化**：先剪枝后量化通常效果更好
   - **剪枝+蒸馏**：使用未剪枝模型作为教师
   - **剪枝+NAS**：自动搜索最优剪枝结构

**实际部署案例分析**：

1. **BERT在移动端的剪枝策略**：
   - 原始模型：110M参数，340MB存储
   - 目标：<50MB，延迟<100ms
   - 策略组合：
     ```
     第1阶段：结构化剪枝
     - 注意力头：12→6 (50%剪枝)
     - FFN维度：3072→1536 (50%剪枝)
     - 层数：12→6 (早退机制)
     结果：参数量降至30M
     
     第2阶段：半结构化剪枝
     - 对剩余权重应用2:4稀疏
     - 微调10个epoch
     结果：等效参数15M
     
     第3阶段：INT8量化
     - 动态量化激活
     - 静态量化权重
     最终：45MB模型，85ms延迟
     ```

2. **GPT-2在边缘设备的优化**：
   - 硬件：Raspberry Pi 4 (4GB RAM)
   - 原始：124M参数，500MB
   - 剪枝方案：
     ```
     层级剪枝率分配：
     - Embedding: 30%通道剪枝
     - Attention: 25%头剪枝
     - FFN: 60%神经元剪枝
     - 输出层: 40%非结构化
     ```
   - 结果：50M参数，200MB，2倍加速

3. **视觉Transformer剪枝**：
   - 模型：ViT-B/16
   - 特殊考虑：
     - 保留class token处理
     - 渐进式patch剪枝
     - 注意力模式分析
   - 混合策略：
     ```
     空间维度：动态token剪枝
     通道维度：结构化剪枝
     注意力：头剪枝+2:4稀疏
     ```

**剪枝模式选择决策树**：
```
是否有专用硬件支持？
├─ 是：检查硬件类型
│   ├─ NVIDIA A100+：优先2:4稀疏
│   ├─ TPU：块稀疏(128×128)
│   └─ 移动NPU：结构化剪枝
└─ 否：通用硬件
    ├─ 精度要求高？
    │   ├─ 是：渐进式结构化剪枝
    │   └─ 否：激进结构化剪枝
    └─ 延迟要求严格？
        ├─ 是：通道/滤波器剪枝
        └─ 否：混合剪枝策略
```

## 9.2 渐进式剪枝策略

渐进式剪枝策略的核心思想是避免一次性大幅度剪枝带来的性能损失，通过逐步增加稀疏度让网络有时间适应结构变化。这类方法在实践中被证明能够达到更高的压缩率同时保持模型性能。

### 9.2.1 渐进幅度剪枝

渐进幅度剪枝（Gradual Magnitude Pruning, GMP）是一种在训练过程中逐步增加稀疏度的方法，最早由Zhu和Gupta在2017年提出，现已成为标准方法之一。

**稀疏度调度函数**：
给定初始稀疏度 $s_0$、目标稀疏度 $s_f$、开始步数 $t_0$ 和结束步数 $t_f$，在步数 $t$ 时的稀疏度为：

$$s_t = s_f + (s_0 - s_f) \left(1 - \frac{t - t_0}{t_f - t_0}\right)^3$$

这里使用三次多项式确保平滑过渡。其他常用的调度函数包括：

1. **线性调度**：
   $$s_t = s_0 + (s_f - s_0) \cdot \frac{t - t_0}{t_f - t_0}$$

2. **余弦调度**：
   $$s_t = s_f + \frac{s_0 - s_f}{2} \left(1 + \cos\left(\pi \cdot \frac{t - t_0}{t_f - t_0}\right)\right)$$

3. **指数调度**：
   $$s_t = s_f - (s_f - s_0) \cdot \exp\left(-\alpha \cdot \frac{t - t_0}{t_f - t_0}\right)$$

**调度函数选择的理论依据**：
不同调度函数对应不同的剪枝哲学：
- **三次多项式**：前期缓慢，中期加速，后期再次放缓，给网络充分适应时间
- **线性调度**：恒定剪枝速率，简单可预测
- **余弦调度**：自然的加速-减速过程，类似学习率调度
- **指数调度**：前期激进，后期保守，适合鲁棒性强的网络

**自适应调度策略**：
基于验证集性能动态调整剪枝速度：
$$s_{t+\Delta t} = s_t + \Delta s \cdot \exp\left(-\beta \cdot \frac{\Delta \mathcal{L}}{\mathcal{L}_{\text{threshold}}}\right)$$

其中 $\Delta \mathcal{L}$ 是验证损失的增加量。当损失增加过快时，自动减缓剪枝速度。

**剪枝频率优化**：
不是每个训练步都更新稀疏模式，而是每隔 $\Delta t$ 步更新一次：

$$\Delta t = \frac{t_f - t_0}{n_{\text{updates}}}$$

其中 $n_{\text{updates}}$ 通常选择100-1000，平衡更新开销和适应性。

**剪枝阈值计算**：
在每个剪枝步骤，计算权重幅度的第 $k$ 个百分位数作为阈值：

$$\theta = \text{Percentile}(|\mathbf{W}|, 100 \cdot s_t)$$

然后更新掩码：

$$M_{ij} = \begin{cases}
1, & \text{if } |W_{ij}| \geq \theta \\
0, & \text{otherwise}
\end{cases}$$

**动量辅助剪枝决策**：
考虑权重的历史变化趋势，使用指数移动平均：

$$\bar{W}_{ij}^{(t)} = \beta \bar{W}_{ij}^{(t-1)} + (1-\beta) |W_{ij}^{(t)}|$$

剪枝决策基于平滑后的权重幅度 $\bar{W}_{ij}$ 而非瞬时值。

**层级同步剪枝**：
为了维持层间平衡，可以采用全局阈值或层级归一化阈值：

1. **全局阈值**：所有层使用相同的剪枝阈值
2. **层级归一化**：
   $$\theta_l = \mu_l + \sigma_l \cdot \Phi^{-1}(s_t)$$
   其中 $\mu_l, \sigma_l$ 是第 $l$ 层权重幅度的均值和标准差。

**与优化器的交互**：
GMP与不同优化器的交互需要特别注意：

- **SGD**：剪枝后需要重置动量缓冲区中被剪枝权重对应的值
- **Adam**：需要同时处理一阶和二阶动量：
  $$m_{ij} \leftarrow m_{ij} \cdot M_{ij}, \quad v_{ij} \leftarrow v_{ij} \cdot M_{ij}$$
- **RMSprop**：类似Adam，需要重置二阶动量

**优化器状态处理的深入分析**：

1. **Adam优化器的特殊考虑**：
   对于被剪枝的权重，其Adam状态需要特殊处理：
   ```
   前向传播：W_ij = 0 (被掩码)
   梯度计算：g_ij ≠ 0 (仍有梯度)
   状态更新：
   - m_ij = β₁m_ij + (1-β₁)g_ij × M_ij
   - v_ij = β₂v_ij + (1-β₂)g²_ij × M_ij
   ```
   
   这防止了被剪枝权重的动量累积。

2. **学习率缩放补偿**：
   剪枝改变了有效参数数量，需要调整学习率：
   $$\eta_{\text{effective}} = \eta \cdot \sqrt{\frac{N_{\text{total}}}{N_{\text{active}}}}$$
   
   其中 $N_{\text{active}} = N_{\text{total}} \cdot (1-s)$。

3. **梯度累积策略**：
   对于大批量训练，梯度累积需要考虑掩码：
   $$\nabla W_{\text{accumulated}} = \sum_{i=1}^{n} \nabla W_i \odot \mathbf{M}$$

**实践中的GMP配置**：

1. **大语言模型的典型配置**：
   ```
   GPT/BERT类模型：
   - 初始稀疏度：s₀ = 0
   - 目标稀疏度：s_f = 0.8-0.9
   - 剪枝开始：10%训练进度
   - 剪枝结束：80%训练进度
   - 更新频率：每100-1000步
   ```

2. **视觉模型的配置**：
   ```
   ResNet/ViT类模型：
   - 初始稀疏度：s₀ = 0.2-0.3
   - 目标稀疏度：s_f = 0.9-0.95
   - 剪枝阶段：占总训练50%
   - 特殊处理：首尾层保守剪枝
   ```

### 9.2.2 迭代剪枝与恢复

迭代剪枝通过多轮"剪枝-恢复训练"循环来达到目标稀疏度。这种方法的理论基础是网络具有自我修复能力，可以通过重新训练补偿部分权重移除带来的影响。

**算法流程**：
1. 初始化：设置剪枝轮数 $N$，每轮剪枝率 $p$
2. 对于 $i = 1, ..., N$：
   - 剪枝：移除当前最小的 $p\%$ 权重
   - 恢复训练：训练 $T$ 个epoch
   - 更新稀疏度：$s_i = 1 - (1-p)^i$

**理论分析**：
假设每轮剪枝后的精度损失为 $\Delta_i$，且通过恢复训练可以恢复 $\alpha \Delta_i$ 的精度（$0 < \alpha < 1$），则总精度损失约为：

$$\Delta_{\text{total}} \approx \sum_{i=1}^N (1-\alpha)^i \Delta_1$$

当 $\alpha$ 接近1时，多轮迭代可以显著减少精度损失。收敛条件为：
$$\Delta_{\text{total}} < \frac{\Delta_1}{1-(1-\alpha)} = \frac{\Delta_1}{\alpha}$$

**自适应恢复训练**：
根据每轮的精度损失动态调整恢复训练的强度：

$$T_i = T_{\text{base}} \cdot \left(1 + \gamma \cdot \frac{\Delta_i}{\Delta_{\text{threshold}}}\right)$$

其中：
- $T_{\text{base}}$：基础训练轮数
- $\gamma$：调节因子
- $\Delta_{\text{threshold}}$：可接受的精度损失阈值

**剪枝率退火策略**：
随着剪枝轮数增加，逐渐降低每轮的剪枝率：

$$p_i = p_{\text{init}} \cdot \exp(-\lambda \cdot i)$$

这种策略在后期更加保守，有助于维持模型性能。

**早停机制**：
监控验证集性能，当连续 $k$ 轮恢复训练无法改善性能时停止：

$$\text{Stop if } \mathcal{L}_{\text{val}}^{(i)} - \mathcal{L}_{\text{val}}^{(i-k)} < \epsilon$$

**批量剪枝 vs. 单元素剪枝**：
- **批量剪枝**：每次移除 $m$ 个权重，计算效率高
- **单元素剪枝**：每次只移除一个权重，更精确但计算开销大

批量大小的选择影响收敛速度和最终性能：
$$m = \max(1, \lfloor p \cdot |\mathbf{W}| / N \rfloor)$$

### 9.2.3 动态稀疏训练

动态稀疏训练（Dynamic Sparse Training, DST）允许在训练过程中改变稀疏模式，这种方法受到神经科学中突触可塑性的启发。代表性工作包括Sparse Evolutionary Training (SET)和Rigged Lottery (RigL)。

**核心思想**：
保持固定的稀疏度，但允许权重连接的拓扑结构动态变化：
- 弱连接被剪除
- 在梯度大的位置生长新连接
- 保持总体稀疏度不变

**权重更新策略**：
1. **剪枝步骤**：移除幅度最小的 $k$ 个权重
2. **生长步骤**：根据梯度信息添加 $k$ 个新连接

**理论基础 - 神经可塑性类比**：
DST模拟了生物神经网络的突触可塑性：
- **突触修剪**：类似大脑发育中的突触消除
- **突触生成**：对应学习过程中的新连接形成
- **稳态可塑性**：保持总连接数相对稳定

数学建模：
$$\frac{d\mathbf{M}}{dt} = \alpha_{\text{grow}} \cdot \mathbf{G} - \alpha_{\text{prune}} \cdot \mathbf{P}$$

其中 $\mathbf{G}$ 是生长信号，$\mathbf{P}$ 是剪枝信号。

**生长策略对比**：

1. **梯度准则（RigL）**：
   $$\text{Score}_{ij} = |\nabla_{W_{ij}} \mathcal{L}| \cdot |\nabla_{a_j} \mathcal{L}|$$
   其中 $\nabla_{a_j} \mathcal{L}$ 是对激活的梯度。

2. **随机生长（SET）**：
   $$P(W_{ij} \text{ grows}) = \frac{1}{|\mathcal{Z}|}$$
   其中 $\mathcal{Z}$ 是所有零权重的集合。

3. **动量准则**：
   $$\text{Score}_{ij} = |m_{ij}|$$
   使用优化器的动量信息指导生长。

4. **二阶准则**：
   $$\text{Score}_{ij} = \frac{(\nabla_{W_{ij}} \mathcal{L})^2}{H_{ij,ij} + \epsilon}$$
   考虑Hessian信息的影响。

**更新调度**：
DST的更新频率随训练进行逐渐降低：

$$f_{\text{update}}(t) = f_0 \cdot \left(1 - \frac{t}{T}\right)^\beta$$

其中 $\beta$ 控制衰减速度，通常取0.5-1.0。

**稀疏模式演化分析**：
定义稀疏模式的Hamming距离：

$$d_H(\mathbf{M}^{(t)}, \mathbf{M}^{(t+1)}) = \frac{1}{mn}\sum_{i,j} |M_{ij}^{(t)} - M_{ij}^{(t+1)}|$$

好的动态稀疏策略应该：
- 训练初期：$d_H$ 较大，探索不同拓扑
- 训练后期：$d_H$ 趋近于0，稳定收敛

**稀疏拓扑的有效性度量**：
定义连接有效性分数：

$$E(\mathbf{M}) = \frac{\sum_{i,j} M_{ij} \cdot |W_{ij}| \cdot |\nabla_{W_{ij}} \mathcal{L}|}{\sum_{i,j} M_{ij}}$$

高分表示当前稀疏拓扑包含了重要的连接。

**与批归一化的交互**：
DST在有BN层的网络中需要特别处理：
- 新生长的连接初始化为小值避免破坏BN统计
- 使用权重标准化：$W_{ij}^{\text{new}} = \epsilon \cdot \text{sign}(\nabla_{W_{ij}} \mathcal{L})$

**收敛性保证**：
在温和条件下（Lipschitz连续梯度），DST可以保证收敛到稳定点：

$$\mathbb{E}[\|\nabla \mathcal{L}(\mathbf{W}^{(T)})\|^2] \leq \frac{2(\mathcal{L}(\mathbf{W}^{(0)}) - \mathcal{L}^*)}{\eta T} + \frac{\eta L \sigma^2}{b}$$

其中 $L$ 是Lipschitz常数，$\sigma^2$ 是梯度方差，$b$ 是批大小。

**DST的实际实现细节**：

1. **稀疏拓扑初始化**：
   - **随机稀疏**：Erdős-Rényi随机图
   - **规则稀疏**：固定度数分布
   - **小世界网络**：局部连接+长程连接
   
   初始化影响收敛速度和最终性能：
   $$P(\text{connection}_{ij}) = \min(1, \frac{c}{\sqrt{n_i \cdot n_j}})$$
   
   其中 $n_i, n_j$ 是层的宽度，$c$ 是常数。

2. **拓扑演化分析**：
   定义拓扑稳定性指标：
   $$\text{Stability}(t) = 1 - \frac{|\mathbf{M}^{(t)} \oplus \mathbf{M}^{(t-\Delta t)}|}{|\mathbf{M}^{(t)}|}$$
   
   其中 $\oplus$ 是异或操作。稳定性接近1表示拓扑已收敛。

3. **层间协调机制**：
   不同层的更新频率应该不同：
   ```
   浅层：频繁更新，探索特征提取
   中层：中等频率，平衡探索与利用
   深层：低频更新，保持语义稳定
   ```

**DST与静态剪枝的性能对比**：

实验表明，在相同稀疏度下：
- DST通常能达到更高精度（+1-3%）
- 训练时间增加20-40%（由于拓扑更新开销）
- 对超参数更敏感，需要仔细调优

**高级DST变体**：

1. **Top-KAST**（Top-K Adaptive Sparse Training）：
   自适应调整每层的稀疏度：
   $$s_l = s_{\text{global}} \cdot \exp\left(-\lambda \cdot \frac{I_l}{\bar{I}}\right)$$
   
   重要层保持更多连接。

2. **GraNet**（Gradient Annealing Network）：
   使用梯度退火控制拓扑变化：
   $$P(\text{rewire}) = \exp\left(-\frac{t}{T_{\text{anneal}}}\right)$$
   
   随训练进行逐渐固定拓扑。

3. **SNFS**（Sparse Networks From Scratch）：
   从头训练稀疏网络，无需预训练密集模型：
   - 使用更大的稀疏网络补偿容量
   - 特殊的初始化策略
   - 修改的优化器适应稀疏梯度

### 9.2.4 学习率调度协同

剪枝过程中的学习率调度对最终性能至关重要。不当的学习率会导致剪枝后的网络无法有效恢复或过早收敛到次优解。

**分段学习率策略**：
- 预热阶段（$t < t_0$）：正常学习率 $\eta_0$
- 剪枝阶段（$t_0 \leq t < t_f$）：递减学习率 $\eta_t = \eta_0 \cdot \cos\left(\frac{\pi(t-t_0)}{2(t_f-t_0)}\right)$
- 微调阶段（$t \geq t_f$）：固定小学习率 $\eta_f = 0.1\eta_0$

**自适应学习率调整**：
基于剪枝引起的梯度范数变化调整学习率：

$$\eta_{\text{adjusted}} = \eta \cdot \frac{\|\nabla \mathcal{L}(\mathbf{W})\|_2}{\|\nabla \mathcal{L}(\mathbf{W}_{\text{pruned}})\|_2}$$

这种调整补偿了剪枝导致的有效学习率变化。

**剪枝感知的学习率策略**：

1. **层级学习率缩放**：
   根据每层的剪枝率调整学习率：
   $$\eta_l = \eta_{\text{base}} \cdot \sqrt{\frac{1}{1-s_l}}$$
   
   这基于有效参数数量减少导致梯度方差增加的理论。

2. **周期性学习率重启**：
   在每次剪枝后重启学习率：
   $$\eta(t) = \eta_{\text{max}} \cdot \frac{1}{2}\left(1 + \cos\left(\frac{\pi \cdot t_{\text{cycle}}}{T_{\text{cycle}}}\right)\right)$$
   
   其中 $t_{\text{cycle}}$ 是当前周期内的步数。

3. **温度缩放学习率**：
   使用温度参数控制学习率衰减速度：
   $$\eta(t) = \eta_0 \cdot \exp\left(-\frac{t}{\tau(s)}\right)$$
   
   其中 $\tau(s) = \tau_0 \cdot (1-s)^{\alpha}$，稀疏度越高，衰减越慢。

**与优化器状态的协调**：

对于带动量的优化器，剪枝时需要特别处理：

1. **动量重置策略**：
   - 完全重置：将被剪枝权重的动量设为0
   - 部分重置：$m_{ij} \leftarrow \gamma \cdot m_{ij}$，其中 $\gamma < 1$
   - 动量转移：将被剪枝权重的动量分配给邻近权重

2. **Adam优化器的特殊处理**：
   $$\begin{aligned}
   m_{ij} &\leftarrow m_{ij} \cdot M_{ij} \\
   v_{ij} &\leftarrow \max(v_{ij} \cdot M_{ij}, \epsilon)
   \end{aligned}$$
   
   避免二阶动量变为0导致的数值不稳定。

**学习率与剪枝频率的关系**：

剪枝频率应与学习率衰减同步：
$$\Delta t_{\text{prune}} = \frac{C}{\eta(t)}$$

其中 $C$ 是常数，确保在学习率高时频繁剪枝，学习率低时减少干扰。

**实验验证的最佳实践**：
- 剪枝期间使用余弦退火学习率
- 剪枝结束后使用常数小学习率微调
- 每次结构化剪枝后短暂提升学习率（学习率spike）
- 对于LLM，剪枝后的学习率通常为预训练峰值学习率的10-30%

**学习率调度的高级技巧**：

1. **剪枝感知的warmup**：
   在剪枝开始前进行特殊的warmup：
   $$\eta_{\text{warmup}}(t) = \eta_{\text{base}} \cdot \frac{t}{T_{\text{warmup}}} \cdot (1 + \alpha \cdot s_{\text{target}})$$
   
   提前让网络适应即将到来的稀疏化。

2. **动态批大小调整**：
   随着稀疏度增加，有效梯度变得更嘈杂，需要增大批大小：
   $$B_{\text{effective}} = B_{\text{base}} \cdot (1 + \beta \cdot s)$$
   
   或等价地降低学习率。

3. **层级异步学习率**：
   不同层使用不同的学习率衰减策略：
   ```
   深层（靠近输出）：快速衰减
   中层：标准衰减
   浅层（靠近输入）：缓慢衰减
   ```
   
   这反映了不同层的剪枝敏感性差异。

**剪枝与正则化的交互**：

剪枝本身是一种隐式正则化，需要调整显式正则化强度：

1. **权重衰减调整**：
   $$\lambda_{\text{wd}}(s) = \lambda_0 \cdot (1-s)^{\gamma}$$
   
   稀疏网络需要更少的权重衰减。

2. **Dropout概率调整**：
   $$p_{\text{dropout}}(s) = p_0 \cdot \exp(-\delta \cdot s)$$
   
   高稀疏度时降低dropout，避免过度正则化。

3. **数据增强策略**：
   稀疏网络容量降低，可能需要：
   - 减少数据增强强度
   - 使用更targeted的增强
   - 考虑知识蒸馏作为软正则化

## 9.3 基于重要性的剪枝准则

选择合适的重要性度量是剪枝成功的关键。理想的重要性准则应该准确反映参数对模型性能的贡献，同时计算效率高。本节将从一阶到二阶，从单元素到结构化，系统介绍各种重要性评估方法。

### 9.3.1 一阶重要性度量

一阶方法基于权重值或梯度信息，计算简单但可能忽略参数间的相互作用。

**权重幅度准则**：
最简单直观的重要性度量是权重的绝对值：

$$I_{ij}^{\text{magnitude}} = |W_{ij}|$$

理论依据：小权重对输出的贡献小。但这忽略了激活值的影响。

**改进的权重幅度准则**：
考虑权重的相对大小：

$$I_{ij}^{\text{normalized}} = \frac{|W_{ij}|}{\sqrt{\sum_k W_{ik}^2}}$$

这种归一化处理了不同层权重尺度差异的问题。

**理论分析 - 为什么幅度剪枝有效**：

1. **隐式正则化视角**：
   训练过程中的权重衰减导致不重要权重自然变小：
   $$\frac{\partial \mathcal{L}_{\text{total}}}{\partial W_{ij}} = \frac{\partial \mathcal{L}}{\partial W_{ij}} + \lambda W_{ij}$$
   
   稳定点处：$|\frac{\partial \mathcal{L}}{\partial W_{ij}}| = \lambda |W_{ij}|$
   
   因此小权重对应小梯度，移除影响小。

2. **信息论视角**：
   权重的信息量可以用其对输出分布的影响衡量：
   $$I(W_{ij}) \approx |W_{ij}| \cdot H(a_j)$$
   
   其中 $H(a_j)$ 是激活的熵。

3. **鲁棒性视角**：
   大权重对应网络的"主干道"，小权重是"备用路径"：
   $$\text{Robustness}(W_{ij}) \propto \frac{|W_{ij}|^2}{\text{Var}(W_{ij})}$$

**梯度幅度准则**：
考虑权重对损失函数的影响：

$$I_{ij}^{\text{gradient}} = |W_{ij} \cdot \nabla_{W_{ij}} \mathcal{L}|$$

这近似于移除权重 $W_{ij}$ 对损失函数的一阶Taylor展开影响。

**移动平均梯度准则**：
使用历史梯度信息增强稳定性：

$$\bar{g}_{ij}^{(t)} = \beta \bar{g}_{ij}^{(t-1)} + (1-\beta) \nabla_{W_{ij}} \mathcal{L}$$
$$I_{ij}^{\text{MA-gradient}} = |W_{ij} \cdot \bar{g}_{ij}|$$

**组合准则**：
结合权重幅度和梯度信息：

$$I_{ij}^{\text{combined}} = |W_{ij}|^\alpha \cdot |\nabla_{W_{ij}} \mathcal{L}|^\beta$$

其中 $\alpha, \beta$ 是超参数，通常 $\alpha + \beta = 1$。常见配置：
- $\alpha = \beta = 0.5$：平衡权重和梯度
- $\alpha = 1, \beta = 0$：退化为幅度准则
- $\alpha = 0, \beta = 1$：纯梯度准则

**激活感知准则**：
考虑前向传播中的激活值：

$$I_{ij}^{\text{activation}} = |W_{ij}| \cdot \mathbb{E}_{\mathbf{x}}[|a_j(\mathbf{x})|]$$

其中 $a_j(\mathbf{x})$ 是输入 $\mathbf{x}$ 对应的第 $j$ 个激活值。

**连接敏感度**：
基于连接的输入输出变化：

$$I_{ij}^{\text{sensitivity}} = \mathbb{E}_{\mathbf{x}}\left[\left|\frac{\partial y_i}{\partial x_j}\right|\right] = |W_{ij}| \cdot \mathbb{E}_{\mathbf{x}}[|a_j'(\mathbf{x})|]$$

**批量梯度统计**：
使用多个批次的梯度统计：

$$I_{ij}^{\text{batch}} = \frac{\mathbb{E}_{\mathcal{B}}[|W_{ij} \cdot \nabla_{W_{ij}} \mathcal{L}|]}{\text{Var}_{\mathcal{B}}[W_{ij} \cdot \nabla_{W_{ij}} \mathcal{L}] + \epsilon}$$

高比值表示稳定且重要的连接。

**高级一阶准则**：

1. **动量感知重要性**：
   利用优化器的动量信息：
   $$I_{ij}^{\text{momentum}} = |W_{ij}| \cdot (1 + \alpha |m_{ij}|)$$
   
   其中 $m_{ij}$ 是动量项。高动量表示该权重正在快速变化。

2. **历史感知重要性**：
   考虑权重的历史轨迹：
   $$I_{ij}^{\text{history}} = \frac{1}{T}\sum_{t=1}^T |W_{ij}^{(t)}| \cdot \exp\left(-\beta \cdot \frac{T-t}{T}\right)$$
   
   近期的权重值权重更高。

3. **相对变化率**：
   $$I_{ij}^{\text{relative}} = \frac{|W_{ij}|}{|W_{ij}^{\text{init}}| + \epsilon} \cdot |\nabla_{W_{ij}} \mathcal{L}|$$
   
   考虑权重相对于初始值的变化。

**计算效率优化**：

1. **采样近似**：
   对于大模型，使用子集估计：
   $$\hat{I}_{ij} = \frac{1}{|\mathcal{S}|}\sum_{s \in \mathcal{S}} |W_{ij} \cdot \nabla_{W_{ij}} \mathcal{L}_s|$$
   
   其中 $\mathcal{S}$ 是数据子集。

2. **分块计算**：
   将权重矩阵分块，并行计算重要性：
   ```
   将W分成k×k块
   对每块并行计算重要性
   合并结果
   ```

3. **增量更新**：
   $$I_{ij}^{(t)} = \rho I_{ij}^{(t-1)} + (1-\rho) I_{ij}^{\text{new}}$$
   
   避免每次从头计算。

### 9.3.2 二阶重要性度量

二阶方法考虑了损失函数的曲率信息，能更准确地预测剪枝的影响，但计算成本较高。

**Taylor展开分析**：
将权重 $W_{ij}$ 设为0后的损失变化可以用Taylor展开近似：

$$\Delta \mathcal{L}_{ij} = -W_{ij} g_{ij} + \frac{1}{2} W_{ij}^2 H_{ij,ij} + O(W_{ij}^3)$$

其中 $g_{ij} = \nabla_{W_{ij}} \mathcal{L}$，$H_{ij,ij}$ 是Hessian矩阵的对角元素。

在最优点附近（$g_{ij} \approx 0$），二阶项主导：
$$\Delta \mathcal{L}_{ij} \approx \frac{1}{2} W_{ij}^2 H_{ij,ij}$$

**Fisher信息近似**：
直接计算Hessian代价高昂，使用Fisher信息矩阵作为近似：

$$F_{ij,ij} = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \left[ \left(\frac{\partial \log p(\mathbf{y}|\mathbf{x}, \mathbf{W})}{\partial W_{ij}}\right)^2 \right]$$

Fisher信息的优势：
- 总是半正定
- 只需要一阶导数
- 可以通过采样高效估计

重要性得分：
$$I_{ij}^{\text{Fisher}} = \frac{1}{2} W_{ij}^2 F_{ij,ij}$$

**经验Fisher信息**：
使用有限样本估计：

$$\hat{F}_{ij,ij} = \frac{1}{N} \sum_{n=1}^N \left(\frac{\partial \mathcal{L}_n}{\partial W_{ij}}\right)^2$$

其中 $\mathcal{L}_n$ 是第 $n$ 个样本的损失。

**最优脑损伤（OBD）准则**：
假设Hessian矩阵是对角的，重要性为：

$$I_{ij}^{\text{OBD}} = \frac{W_{ij}^2}{2[H^{-1}]_{ij,ij}}$$

这考虑了参数的不确定性：Hessian对角元素大表示该参数对损失敏感。

**最优脑外科（OBS）准则**：
考虑完整的Hessian矩阵，允许其他权重补偿被剪枝权重：

$$\Delta \mathbf{W} = -\frac{W_{ij}}{[H^{-1}]_{ij,ij}} H^{-1}_{\cdot,ij}$$

剪枝 $W_{ij}$ 后的损失增加：
$$\Delta \mathcal{L} = \frac{W_{ij}^2}{2[H^{-1}]_{ij,ij}}$$

**块对角Hessian近似**：
为了平衡精度和计算效率，使用块对角近似：

$$\mathbf{H} = \begin{bmatrix}
\mathbf{H}_1 & \mathbf{0} & \cdots \\
\mathbf{0} & \mathbf{H}_2 & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}$$

每个块对应一层或一组相关参数。

**Kronecker因子化近似（K-FAC）**：
对于全连接层 $\mathbf{y} = \mathbf{W}\mathbf{x}$，Fisher信息可以近似为：

$$\mathbf{F} \approx \mathbb{E}[\mathbf{x}\mathbf{x}^T] \otimes \mathbb{E}[\mathbf{g}\mathbf{g}^T]$$

其中 $\mathbf{g}$ 是输出的梯度，$\otimes$ 是Kronecker积。

**Woodbury矩阵恒等式优化**：
对于低秩更新的Hessian逆：

$$(\mathbf{H} + \mathbf{U}\mathbf{V}^T)^{-1} = \mathbf{H}^{-1} - \mathbf{H}^{-1}\mathbf{U}(\mathbf{I} + \mathbf{V}^T\mathbf{H}^{-1}\mathbf{U})^{-1}\mathbf{V}^T\mathbf{H}^{-1}$$

这允许高效更新剪枝后的Hessian逆。

**二阶方法的实践考虑**：

1. **Hessian近似的权衡**：
   ```
   方法              | 计算复杂度 | 内存需求 | 精度
   ----------------|-----------|---------|------
   完整Hessian      | O(n³)     | O(n²)   | 最高
   对角Hessian      | O(n)      | O(n)    | 低
   块对角Hessian    | O(kb³)    | O(kb²)  | 中
   K-FAC           | O(n^1.5)  | O(n)    | 中高
   ```

2. **自适应Hessian计算**：
   根据层的重要性分配计算资源：
   $$\text{BlockSize}_l = \min(n_l, \lfloor B_{\text{base}} \cdot \exp(-\alpha \cdot \text{depth}_l) \rfloor)$$
   
   深层使用更小的块以节省计算。

3. **Hessian的数值稳定性**：
   添加正则化项确保可逆性：
   $$\tilde{\mathbf{H}} = \mathbf{H} + \lambda \mathbf{I}$$
   
   其中 $\lambda = \max(\epsilon, \alpha \cdot \text{tr}(\mathbf{H})/n)$。

**混合准则策略**：

结合一阶和二阶信息：
$$I_{ij}^{\text{hybrid}} = \alpha \cdot I_{ij}^{\text{first-order}} + (1-\alpha) \cdot I_{ij}^{\text{second-order}}$$

其中 $\alpha$ 可以根据计算预算动态调整：
- 计算资源充足：$\alpha = 0.2$（偏重二阶）
- 计算资源有限：$\alpha = 0.8$（偏重一阶）
- 自适应：$\alpha = \exp(-\beta \cdot \text{budget})$

### 9.3.3 结构化重要性度量

对于结构化剪枝，需要评估整个结构单元（如通道、滤波器、注意力头）的重要性。这比单个权重的评估更复杂，需要考虑结构内部的协同作用。

**通道重要性聚合**：
对于第 $c$ 个通道，聚合所有相关权重的重要性：

$$I_c^{\text{channel}} = \sum_{i,j,k} I_{c,i,j,k}$$

更精细的聚合策略：
- **L2范数**：$I_c = \|\mathbf{W}_c\|_2 = \sqrt{\sum_{i,j,k} W_{c,i,j,k}^2}$
- **L1范数**：$I_c = \|\mathbf{W}_c\|_1 = \sum_{i,j,k} |W_{c,i,j,k}|$
- **L∞范数**：$I_c = \|\mathbf{W}_c\|_\infty = \max_{i,j,k} |W_{c,i,j,k}|$

**基于特征图的度量**：
使用批归一化的缩放因子作为通道重要性指标：

$$I_c^{\text{BN}} = |\gamma_c|$$

其中 $\gamma_c$ 是批归一化层的缩放参数。理论依据：
- BN层学习每个通道的重要性权重
- 小的 $\gamma_c$ 表示该通道输出经常被缩小
- 可以直接用于网络瘦身（Network Slimming）

**激活统计度量**：
基于特征图的统计信息：

$$I_c^{\text{activation}} = \mathbb{E}_{\mathbf{x}} \left[ \|\mathbf{a}_c(\mathbf{x})\|_2 \right]$$

扩展的激活度量：
- **方差**：$I_c^{\text{var}} = \text{Var}_{\mathbf{x}}[\mathbf{a}_c(\mathbf{x})]$
- **熵**：$I_c^{\text{entropy}} = -\sum_i p_i \log p_i$，其中 $p_i$ 是激活值的分布
- **稀疏度**：$I_c^{\text{sparse}} = 1 - \frac{\mathbb{E}[|\mathbf{a}_c|]}{\mathbb{E}[\mathbf{a}_c^2]^{1/2}}$

**梯度流度量**：
考虑反向传播中的梯度流：

$$I_c^{\text{grad-flow}} = \mathbb{E}_{\mathbf{x}} \left[ \left\| \frac{\partial \mathcal{L}}{\partial \mathbf{a}_c} \right\|_2 \right]$$

**几何中值度量**：
对于滤波器剪枝，使用几何中值找到最"冗余"的滤波器：

$$i^* = \arg\min_i \sum_{j \neq i} \|\mathbf{W}_i - \mathbf{W}_j\|_2$$

距离其他滤波器最近的被认为是冗余的。

**注意力头重要性**：
对于Transformer中的多头注意力：

$$I_h^{\text{attention}} = \mathbb{E}_{\mathbf{x}} \left[ \|\text{Attention}_h(\mathbf{Q}, \mathbf{K}, \mathbf{V})\|_F \right]$$

考虑注意力模式的多样性：
$$I_h^{\text{diversity}} = -\text{tr}(\mathbf{A}_h \log \mathbf{A}_h)$$

其中 $\mathbf{A}_h$ 是第 $h$ 个头的平均注意力矩阵。

**层级重要性评估**：
对于整层剪枝，需要全局视角：

$$I_l^{\text{layer}} = \frac{\|\mathbf{F}_l - \mathbf{F}_{l-1}\|_F}{\|\mathbf{F}_{l-1}\|_F}$$

其中 $\mathbf{F}_l$ 是第 $l$ 层的输出特征。

**结构化Taylor展开**：
将Taylor展开推广到结构化情况：

$$\Delta \mathcal{L}_{\mathcal{S}} = \sum_{i \in \mathcal{S}} W_i g_i + \frac{1}{2} \sum_{i,j \in \mathcal{S}} W_i H_{ij} W_j$$

其中 $\mathcal{S}$ 是被剪枝的结构单元（如一个通道的所有权重）。

**组LASSO正则化诱导的重要性**：
使用组LASSO训练时自然产生的稀疏性：

$$\mathcal{R}_{\text{group}} = \lambda \sum_{g} \|\mathbf{W}_g\|_2$$

训练后，$\|\mathbf{W}_g\|_2$ 直接反映组的重要性。

**高级结构化重要性度量**：

1. **互信息准则**：
   评估通道与输出的互信息：
   $$I_c^{\text{MI}} = I(\mathbf{a}_c; \mathbf{y}) = \mathbb{E}\left[\log\frac{p(\mathbf{a}_c, \mathbf{y})}{p(\mathbf{a}_c)p(\mathbf{y})}\right]$$
   
   实践中使用MINE (Mutual Information Neural Estimation)近似。

2. **因果重要性**：
   通过干预实验评估因果效应：
   $$I_c^{\text{causal}} = \mathbb{E}[\mathcal{L}(\mathbf{y}|\text{do}(\mathbf{a}_c = 0))] - \mathbb{E}[\mathcal{L}(\mathbf{y})]$$
   
   其中 $\text{do}(\cdot)$ 表示因果干预。

3. **Shapley值方法**：
   基于合作博弈论的公平贡献分配：
   $$\phi_c = \sum_{S \subseteq N \setminus \{c\}} \frac{|S|!(|N|-|S|-1)!}{|N|!}[v(S \cup \{c\}) - v(S)]$$
   
   其中 $v(S)$ 是子集 $S$ 的模型性能。

**特定架构的重要性度量**：

1. **Transformer专用**：
   - **注意力熵**：
     $$I_h^{\text{entropy}} = -\sum_{i,j} A_{h,ij} \log A_{h,ij}$$
     
     低熵表示注意力集中，可能更重要。
   
   - **查询-键相似度分散度**：
     $$I_h^{\text{QK}} = \text{std}(\mathbf{Q}_h \mathbf{K}_h^T)$$
     
     高分散度表示区分能力强。

2. **CNN专用**：
   - **感受野有效性**：
     $$I_c^{\text{RF}} = \frac{\|\nabla_{\mathbf{x}} \mathbf{a}_c\|_0}{\text{RF}_{\text{theoretical}}}$$
     
     实际使用的感受野比例。
   
   - **特征图稀疏度**：
     $$I_c^{\text{sparse}} = 1 - \frac{\|\mathbf{a}_c\|_1}{\|\mathbf{a}_c\|_0 \cdot \|\mathbf{a}_c\|_\infty}$$

**计算优化技巧**：

1. **重要性缓存机制**：
   ```python
   importance_cache = {}
   update_frequency = {
       "shallow": 10,   # 浅层频繁更新
       "middle": 50,    # 中层适度更新  
       "deep": 100      # 深层稀疏更新
   }
   ```

2. **并行化策略**：
   - 通道级并行：不同通道独立计算
   - 批次级并行：不同数据批次并行
   - 层级流水线：前后层重叠计算

3. **早停机制**：
   当重要性排序稳定时停止计算：
   $$\text{Kendall-}\tau(R^{(t)}, R^{(t-1)}) > \theta$$

### 9.3.4 彩票假设与剪枝

彩票假设（Lottery Ticket Hypothesis）由Frankle和Carbin在2018年提出，这一发现深刻影响了对神经网络剪枝和稀疏性的理解。

**核心假设**：
随机初始化的密集网络包含一个子网络（"中奖彩票"），当独立训练时可以达到原网络的精度，甚至更快收敛。

**迭代幅度剪枝（IMP）算法**：
1. 随机初始化网络 $f(\mathbf{x}; \mathbf{W}_0)$，保存初始权重
2. 训练网络至收敛，得到 $\mathbf{W}_T$
3. 剪枝 $p\%$ 最小幅度权重，得到掩码 $\mathbf{M}$
4. 将剩余权重重置为初始值：$\mathbf{W} = \mathbf{W}_0 \odot \mathbf{M}$
5. 重复步骤2-4直到达到目标稀疏度

**学习率回退技巧**：
对于大规模网络（如ResNet-50、BERT），需要回退到训练早期的权重：

$$\mathbf{W}_{\text{rewind}} = \mathbf{W}_k \odot \mathbf{M}, \quad k \ll T$$

通常 $k$ 选择为总训练步数的1-10%。这被称为"晚期重置"（Late Resetting）。

**理论解释**：

1. **优化景观视角**：
   - 好的稀疏子网络对应于损失景观中的"盆地"
   - 适当的初始化使得这些子网络更容易通过梯度下降找到
   - 密集网络提供了多条到达良好最小值的路径

2. **信息论视角**：
   - 剪枝过程是一种信息压缩
   - "中奖彩票"包含了解决任务的必要信息
   - 初始化提供了正确的归纳偏置

3. **隐式正则化视角**：
   - 稀疏结构本身是一种正则化
   - 限制了模型的表达能力，防止过拟合

**强彩票假设**：
不仅存在可以训练到相同精度的稀疏子网络，而且存在无需训练就能达到良好性能的子网络。这导致了"边缘弹出"（Edge Popup）算法：

$$\mathbf{M} = \mathbb{1}[\text{score}(\mathbf{W}_0) > \theta]$$

其中score函数可以是权重幅度、梯度等。

**彩票假设的变体**：

1. **多重彩票**：一个网络可能包含多个不同的"中奖彩票"
2. **迁移彩票**：在一个任务上找到的彩票可以迁移到相关任务
3. **弹性彩票**：彩票对小的权重扰动具有鲁棒性

**实验发现的规律**：
- **临界稀疏度**：存在一个临界点，超过后性能急剧下降
- **网络深度影响**：深网络的彩票更难找到
- **任务复杂度**：复杂任务需要更密集的彩票

**与传统剪枝的区别**：
- 传统剪枝：训练密集网络 → 剪枝 → 微调
- 彩票假设：识别子网络 → 重新初始化 → 从头训练

**实际应用考虑**：
1. **计算成本**：IMP需要多轮完整训练，成本高
2. **稀疏度限制**：实际应用中很难达到极高稀疏度
3. **硬件支持**：找到的彩票可能不适合硬件加速

**彩票假设的最新进展**：

1. **Early-Bird票据**：
   在训练早期就能识别出中奖彩票：
   $$\text{EB-Score}(t) = \frac{1}{L}\sum_{l=1}^L \text{Hamming}(\mathbf{M}_l^{(t)}, \mathbf{M}_l^{(t-\Delta t)})$$
   
   当EB-Score低于阈值时，掩码已经稳定。

2. **多任务彩票**：
   一个彩票可能在多个相关任务上都表现良好：
   $$\mathbf{M}^* = \arg\min_{\mathbf{M}} \sum_{k=1}^K \alpha_k \mathcal{L}_k(\mathbf{W}_0 \odot \mathbf{M})$$
   
   其中 $\mathcal{L}_k$ 是第 $k$ 个任务的损失。

3. **连续稀疏化**：
   使用连续松弛替代离散掩码：
   $$\mathbf{W}_{\text{sparse}} = \mathbf{W} \odot \sigma(\mathbf{S}/\tau)$$
   
   其中 $\mathbf{S}$ 是可学习的分数，$\tau$ 是温度参数。

**彩票假设的理论解释新进展**：

1. **NTK (Neural Tangent Kernel)视角**：
   稀疏网络的NTK可能与密集网络相似：
   $$\mathbf{K}_{\text{sparse}} \approx \mathbf{K}_{\text{dense}}$$
   
   这解释了为什么稀疏网络能达到相似性能。

2. **模式连通性**：
   好的彩票位于损失景观的同一盆地：
   $$\exists \gamma: [0,1] \rightarrow \Theta, \mathcal{L}(\gamma(t)) \leq \max\{\mathcal{L}(\gamma(0)), \mathcal{L}(\gamma(1))\} + \epsilon$$

3. **信息瓶颈理论**：
   剪枝过程是一种信息压缩：
   $$I(X; T) - \beta I(T; Y)$$
   
   其中 $T$ 是稀疏表示，平衡压缩和预测性能。

**实用彩票搜索算法**：

1. **渐进式彩票搜索**：
   ```
   初始化：dense_model, sparsity_targets = [0.2, 0.4, 0.6, 0.8]
   对每个sparsity in sparsity_targets:
       训练当前模型
       识别重要权重
       创建掩码
       如果性能下降 > 阈值:
           回退到上一个sparsity
           使用更保守的剪枝
   ```

2. **并行彩票搜索**：
   同时探索多个剪枝路径：
   $$\{\mathbf{M}_1, \mathbf{M}_2, ..., \mathbf{M}_k\} = \text{ParallelSearch}(\mathbf{W}_0)$$
   
   选择验证性能最好的路径。

3. **元学习彩票**：
   学习如何为新任务快速找到彩票：
   $$\mathbf{M}_{\text{new}} = f_\theta(\mathcal{D}_{\text{task}}, \mathbf{W}_0)$$
   
   其中 $f_\theta$ 是元学习的剪枝策略。

## 9.4 剪枝后的微调技术

剪枝后的微调是恢复和提升模型性能的关键步骤。不当的微调策略可能导致剪枝的努力白费，而精心设计的微调方案能够让稀疏模型达到甚至超越原始密集模型的性能。

### 9.4.1 学习率回退与微调

剪枝后的微调需要特别的学习率策略，因为网络的有效容量和梯度流都发生了变化。

**学习率回退（Learning Rate Rewinding）**：
将学习率重置到训练过程中的某个早期状态：

$$\eta_{\text{rewind}} = \eta(t_{\text{rewind}})$$

其中 $t_{\text{rewind}}$ 通常选择为原始训练的10-20%位置。

回退策略的选择：
- **线性回退**：$t_{\text{rewind}} = T \cdot (1 - s)$，稀疏度越高，回退越早
- **对数回退**：$t_{\text{rewind}} = T \cdot \exp(-k \cdot s)$
- **自适应回退**：基于验证集性能动态确定

**理论基础**：
学习率回退的有效性可以从优化景观角度理解：

1. **损失景观变化**：
   剪枝改变了损失景观的几何结构：
   $$\mathcal{L}_{\text{pruned}}(\mathbf{W}) = \mathcal{L}(\mathbf{W} \odot \mathbf{M})$$
   
   新景观可能有不同的曲率和局部最小值分布。

2. **有效学习率理论**：
   剪枝后的有效学习率需要考虑参数减少：
   $$\eta_{\text{effective}} = \eta \cdot \frac{\|\nabla_{\mathbf{W}} \mathcal{L}\|_2}{\|\nabla_{\mathbf{W} \odot \mathbf{M}} \mathcal{L}\|_2}$$
   
   通常需要提高学习率以补偿梯度稀疏性。

3. **临界学习率**：
   稳定训练的最大学习率：
   $$\eta_{\text{critical}} = \frac{2}{\lambda_{\text{max}}(\mathbf{H}_{\text{pruned}})}$$
   
   其中 $\lambda_{\text{max}}$ 是剪枝后Hessian的最大特征值。

**分层学习率策略**：
对于不同稀疏度的层使用不同的学习率：

$$\eta_l = \eta_{\text{base}} \cdot (1 - s_l)^\alpha$$

其中 $s_l$ 是第 $l$ 层的稀疏度，$\alpha > 0$ 是缩放因子。

理论依据：
- 稀疏层的有效参数少，需要更大的学习率
- 梯度在稀疏结构中的传播不同于密集网络

**循环学习率策略**：
使用周期性学习率帮助逃离次优局部最小值：

$$\eta(t) = \eta_{\text{min}} + \frac{\eta_{\text{max}} - \eta_{\text{min}}}{2} \left(1 + \cos\left(\frac{\pi t}{T_{\text{cycle}}}\right)\right)$$

每个周期开始时的高学习率有助于探索新的参数空间。

**梯度掩码技术**：
确保被剪枝的权重在微调过程中保持为零：

$$\mathbf{W}^{(t+1)} = \mathbf{W}^{(t)} - \eta \cdot (\nabla_{\mathbf{W}} \mathcal{L} \odot \mathbf{M})$$

扩展的梯度处理：
- **梯度裁剪**：$\hat{g} = \text{clip}(g, -\theta, \theta)$，防止梯度爆炸
- **梯度归一化**：$\hat{g} = g / \|g\|_2$，稳定训练
- **自适应梯度缩放**：根据层的稀疏度调整梯度

**微调阶段设计**：

1. **快速恢复阶段**（0-25% epochs）：
   - 高学习率：$\eta = 0.5 \eta_{\text{original}}$
   - 目标：快速适应新结构

2. **精细调整阶段**（25-75% epochs）：
   - 逐渐降低学习率
   - 使用余弦退火或阶梯式衰减

3. **收敛阶段**（75-100% epochs）：
   - 极小学习率：$\eta = 0.01 \eta_{\text{original}}$
   - 目标：精确收敛到最优点

**早停与检查点策略**：
- 监控验证集性能，避免过拟合
- 保存多个检查点，选择最佳模型
- 使用移动平均模型提升稳定性

**高级微调技巧**：

1. **梯度手术（Gradient Surgery）**：
   修正剪枝引起的梯度冲突：
   $$\tilde{g}_i = g_i - \frac{g_i^T g_j}{\|g_j\|^2} g_j$$
   
   当层间梯度冲突时，投影到正交空间。

2. **弹性权重巩固（EWC）**：
   保护重要权重不被过度更新：
   $$\mathcal{L}_{\text{EWC}} = \mathcal{L} + \frac{\lambda}{2} \sum_i F_i (W_i - W_i^*)^2$$
   
   其中 $F_i$ 是Fisher信息矩阵对角元素。

3. **路径积分正则化**：
   保持剪枝前后的函数行为相似：
   $$\mathcal{L}_{\text{path}} = \mathbb{E}_{\mathbf{x}} \left[\|\mathbf{f}_{\text{pruned}}(\mathbf{x}) - \mathbf{f}_{\text{dense}}(\mathbf{x})\|^2\right]$$

**微调阶段的数据策略**：

1. **硬样本挖掘**：
   重点微调在剪枝后性能下降最大的样本：
   $$\mathcal{D}_{\text{hard}} = \{(\mathbf{x}, y) : \Delta\mathcal{L}(\mathbf{x}, y) > \theta\}$$

2. **课程学习**：
   从简单样本逐渐过渡到困难样本：
   $$p_t(\mathbf{x}) \propto \exp(-\lambda_t \cdot \text{difficulty}(\mathbf{x}))$$
   
   其中 $\lambda_t$ 随时间递减。

3. **数据增强调整**：
   ```
   剪枝初期：减少增强强度，帮助稳定
   剪枝中期：恢复正常增强
   剪枝后期：增加增强强度，提升泛化
   ```

### 9.4.2 知识蒸馏辅助微调

使用原始密集模型作为教师模型，指导剪枝模型的微调，这种方法能够有效传递密集模型的"暗知识"（dark knowledge）到稀疏模型。

**蒸馏损失函数**：
$$\mathcal{L}_{\text{distill}} = (1-\lambda) \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{KD}}$$

其中知识蒸馏损失为：

$$\mathcal{L}_{\text{KD}} = \tau^2 \cdot \text{KL}(p_{\text{student}}(\mathbf{y}/\tau | \mathbf{x}) \| p_{\text{teacher}}(\mathbf{y}/\tau | \mathbf{x}))$$

$\tau$ 是温度参数，通常设为3-5。高温度使得软标签更平滑，传递更多信息。

**温度调度策略**：
动态调整温度参数以优化知识传递：

$$\tau(t) = \tau_{\text{max}} - (\tau_{\text{max}} - \tau_{\text{min}}) \cdot \frac{t}{T}$$

早期使用高温度传递全局知识，后期降低温度关注硬预测。

**特征蒸馏**：
除了输出层的蒸馏，还可以对中间层特征进行蒸馏：

$$\mathcal{L}_{\text{feature}} = \sum_{l \in \mathcal{L}_{\text{distill}}} \|\mathbf{F}_l^{\text{student}} - \mathbf{F}_l^{\text{teacher}}\|_2^2$$

**注意力蒸馏**：
对于Transformer模型，蒸馏注意力图：

$$\mathcal{L}_{\text{attention}} = \sum_{h=1}^H \text{MSE}(\mathbf{A}_h^{\text{student}}, \mathbf{A}_h^{\text{teacher}})$$

其中 $\mathbf{A}_h$ 是第 $h$ 个注意力头的注意力矩阵。

**自适应蒸馏权重**：
根据剪枝程度动态调整蒸馏权重：

$$\lambda_l = \lambda_0 \cdot \exp(k \cdot s_l)$$

其中 $k > 0$ 控制随稀疏度增加的程度。稀疏度越高的层需要更多的指导。

**渐进式蒸馏**：
分阶段调整蒸馏目标：

1. **初始阶段**：完全模仿教师
   $$\mathcal{L} = \mathcal{L}_{\text{KD}}$$

2. **过渡阶段**：平衡任务损失和蒸馏损失
   $$\mathcal{L} = 0.5\mathcal{L}_{\text{task}} + 0.5\mathcal{L}_{\text{KD}}$$

3. **最终阶段**：主要优化任务性能
   $$\mathcal{L} = 0.9\mathcal{L}_{\text{task}} + 0.1\mathcal{L}_{\text{KD}}$$

**对比蒸馏**：
使用对比学习增强特征对齐：

$$\mathcal{L}_{\text{contrast}} = -\log \frac{\exp(\text{sim}(f_s, f_t)/\tau)}{\sum_{i=1}^N \exp(\text{sim}(f_s, f_i)/\tau)}$$

其中 $f_s, f_t$ 分别是学生和教师的特征表示。

**蒸馏与剪枝的协同优化**：
同时进行剪枝和蒸馏：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{distill}} + \beta \mathcal{R}_{\text{sparsity}}$$

其中 $\mathcal{R}_{\text{sparsity}}$ 是促进稀疏的正则项。

**高级蒸馏技术**：

1. **关系知识蒸馏（RKD）**：
   不仅蒸馏个体输出，还蒸馏样本间的关系：
   $$\mathcal{L}_{\text{RKD}} = \|\mathbf{R}^{\text{student}} - \mathbf{R}^{\text{teacher}}\|_F^2$$
   
   其中 $R_{ij} = \frac{\langle f_i, f_j \rangle}{\|f_i\| \|f_j\|}$ 是特征间的余弦相似度。

2. **流形蒸馏**：
   保持数据在特征空间的流形结构：
   $$\mathcal{L}_{\text{manifold}} = \sum_{i,j} w_{ij} \|d_{ij}^{\text{student}} - d_{ij}^{\text{teacher}}\|^2$$
   
   其中 $d_{ij}$ 是特征空间中的距离。

3. **不确定性感知蒸馏**：
   根据教师模型的不确定性调整蒸馏权重：
   $$w(\mathbf{x}) = \exp(-\alpha \cdot H(p_{\text{teacher}}(\mathbf{y}|\mathbf{x})))$$
   
   高熵（不确定）的预测获得较低权重。

**层间蒸馏策略**：

1. **深度匹配**：
   选择性地匹配不同深度的层：
   $$\mathcal{L}_{\text{depth}} = \sum_{(i,j) \in \mathcal{P}} \|\mathbf{F}_i^{\text{student}} - \phi(\mathbf{F}_j^{\text{teacher}})\|^2$$
   
   其中 $\mathcal{P}$ 是层对匹配，$\phi$ 是适配函数。

2. **注意力转移**：
   专门针对注意力机制的蒸馏：
   $$\mathcal{L}_{\text{attention}} = \sum_h \text{KL}(A_h^{\text{student}} \| A_h^{\text{teacher}})$$

3. **梯度匹配**：
   匹配关于输入的梯度：
   $$\mathcal{L}_{\text{gradient}} = \|\nabla_{\mathbf{x}} f^{\text{student}} - \nabla_{\mathbf{x}} f^{\text{teacher}}\|^2$$

### 9.4.3 剪枝感知训练

剪枝感知训练（Pruning-Aware Training）在训练过程中就考虑剪枝的影响。

**软剪枝方法**：
使用可学习的掩码参数：

$$\mathbf{W}_{\text{effective}} = \mathbf{W} \odot \sigma(\mathbf{S})$$

其中 $\mathbf{S}$ 是可学习的得分矩阵，$\sigma$ 是sigmoid函数。

**直通估计器（STE）**：
在前向传播时使用硬阈值，反向传播时使用软阈值：

前向：
$$\mathbf{M} = \mathbb{1}[\sigma(\mathbf{S}) > 0.5]$$

反向：
$$\frac{\partial \mathcal{L}}{\partial \mathbf{S}} = \frac{\partial \mathcal{L}}{\partial \mathbf{W}_{\text{effective}}} \odot \mathbf{W} \odot \sigma'(\mathbf{S})$$

**稀疏正则化**：
添加促进稀疏的正则项：

$$\mathcal{L}_{\text{sparse}} = \beta \sum_{i,j} \sigma(S_{ij})(1 - \sigma(S_{ij}))$$

这个正则项在 $\sigma(S_{ij}) = 0.5$ 时达到最大值，推动参数向0或1移动。

### 9.4.4 渐进式知识转移

对于大规模语言模型，可以采用渐进式的知识转移策略。

**层级蒸馏策略**：
1. 首先剪枝和微调底层（接近输入）
2. 逐步向上层推进
3. 每层微调时冻结已完成的层

**块级别剪枝与恢复**：
对于Transformer模型，以块为单位进行剪枝：

$$\text{Block}_i^{\text{pruned}} = \text{Prune}(\text{Block}_i^{\text{dense}}, s_i)$$

每个块的微调目标：

$$\min_{\theta_i^{\text{pruned}}} \|\text{Block}_i^{\text{pruned}}(\mathbf{X}) - \text{Block}_i^{\text{dense}}(\mathbf{X})\|_F^2$$

**动态容量分配**：
根据层的重要性动态分配剪枝率：

$$s_l = s_{\text{avg}} + \alpha \cdot \frac{I_l - \bar{I}}{\text{std}(I)}$$

其中 $I_l$ 是第 $l$ 层的重要性得分。

**大模型剪枝的特殊考虑**：

1. **记忆保护机制**：
   LLM中某些权重编码了重要的事实知识：
   $$\mathcal{L}_{\text{memory}} = \sum_{(\mathbf{x}, y) \in \mathcal{D}_{\text{facts}}} \mathcal{L}(f(\mathbf{x}), y)$$
   
   其中 $\mathcal{D}_{\text{facts}}$ 是关键事实数据集。

2. **能力分解剪枝**：
   将模型能力分解，分别优化：
   ```
   语言理解能力：保守剪枝
   推理能力：中等剪枝
   创造性能力：可激进剪枝
   ```

3. **增量剪枝与检查点**：
   ```
   for epoch in range(num_epochs):
       if epoch % checkpoint_freq == 0:
           保存当前模型
           评估各项能力指标
           if 性能下降超过阈值:
               回滚到上一检查点
               调整剪枝策略
   ```

**混合精度剪枝**：

结合剪枝和量化，不同重要性的层使用不同精度：

$$\text{Precision}_l = \begin{cases}
\text{FP16} & \text{if } I_l > \theta_{\text{high}} \\
\text{INT8} & \text{if } \theta_{\text{low}} < I_l \leq \theta_{\text{high}} \\
\text{INT4} & \text{if } I_l \leq \theta_{\text{low}}
\end{cases}$$

**端到端优化流程**：

1. **预分析阶段**：
   - 敏感性分析确定各层容忍度
   - 硬件性能建模预测加速比
   - 设定多目标优化目标

2. **迭代优化阶段**：
   ```
   while not converged:
       # 剪枝步骤
       更新重要性得分
       执行剪枝
       
       # 恢复步骤
       知识蒸馏微调
       评估性能指标
       
       # 调整步骤
       if 精度损失 > 容忍度:
           降低剪枝率
       if 加速比 < 目标:
           增加剪枝率
   ```

3. **后处理阶段**：
   - 结构优化（如通道对齐）
   - 图优化（算子融合）
   - 部署格式转换

## 本章小结

本章深入探讨了模型剪枝的核心技术和实践方法：

1. **剪枝模式选择**：
   - 非结构化剪枝：最高压缩率，但需要专门硬件支持
   - 结构化剪枝：硬件友好，可直接在通用设备上加速
   - 半结构化剪枝：在新型硬件上的折中方案

2. **渐进式剪枝策略**：
   - 稀疏度调度：$s_t = s_f + (s_0 - s_f)(1 - \frac{t-t_0}{t_f-t_0})^3$
   - 迭代剪枝可以显著减少精度损失
   - 动态稀疏训练允许网络自适应调整结构

3. **重要性评估准则**：
   - 一阶准则：权重幅度、梯度幅度
   - 二阶准则：Fisher信息、最优脑损伤
   - 结构化准则：通道重要性、激活统计

4. **微调技术**：
   - 学习率回退是关键技巧
   - 知识蒸馏可以有效恢复性能
   - 剪枝感知训练提供端到端优化

关键公式回顾：
- Taylor展开重要性：$\Delta \mathcal{L}_{ij} = -W_{ij} g_{ij} + \frac{1}{2} W_{ij}^2 H_{ij,ij}$
- 知识蒸馏损失：$\mathcal{L} = (1-\lambda) \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{KD}}$
- 软剪枝掩码：$\mathbf{W}_{\text{effective}} = \mathbf{W} \odot \sigma(\mathbf{S})$

## 练习题

### 基础题

1. **稀疏度计算**
   给定一个 $4 \times 4$ 的权重矩阵，其中有6个元素被剪枝为0。计算该矩阵的稀疏度。
   
   *Hint*: 稀疏度是零元素占总元素的比例。

2. **结构化剪枝影响**
   一个卷积层的权重形状为 $(64, 128, 3, 3)$（输出通道，输入通道，核高，核宽）。如果剪枝50%的输出通道，剪枝后的参数量是原来的多少？
   
   *Hint*: 考虑剪枝如何改变权重张量的形状。

3. **渐进剪枝计算**
   使用渐进幅度剪枝公式，计算当 $s_0 = 0$, $s_f = 0.9$, $t_0 = 1000$, $t_f = 10000$ 时，在 $t = 5500$ 步的目标稀疏度。
   
   *Hint*: 直接代入公式 $s_t = s_f + (s_0 - s_f)(1 - \frac{t-t_0}{t_f-t_0})^3$。

4. **2:4稀疏模式**
   对于向量 $[0.1, -0.5, 0.3, -0.2]$，找出满足2:4稀疏模式且使得剩余元素的L2范数最大的剪枝方案。
   
   *Hint*: 2:4模式要求每4个元素中保留2个，选择绝对值最大的。

### 挑战题

5. **最优剪枝阈值**
   证明对于权重服从正态分布 $\mathcal{N}(0, \sigma^2)$ 的情况，要达到稀疏度 $s$，最优的幅度剪枝阈值为 $\theta = \sigma \cdot \Phi^{-1}((1+s)/2)$，其中 $\Phi^{-1}$ 是标准正态分布的逆累积分布函数。
   
   *Hint*: 利用正态分布的对称性和累积分布函数的性质。

6. **Hessian对角近似误差**
   考虑二次损失函数 $\mathcal{L}(\mathbf{w}) = \frac{1}{2}\mathbf{w}^T\mathbf{H}\mathbf{w}$，其中 $\mathbf{H}$ 是对称正定矩阵。分析当使用对角近似 $\mathbf{H}_{\text{diag}} = \text{diag}(\mathbf{H})$ 时，剪枝决策的误差界。
   
   *Hint*: 考虑非对角元素对重要性评估的影响。

7. **动态稀疏训练收敛性**
   设计一个动态稀疏训练的更新规则，使得稀疏模式的Hamming距离 $d_H(\mathbf{M}^{(t)}, \mathbf{M}^{(t+1)})$ 随训练进行单调递减。讨论这种策略的优缺点。
   
   *Hint*: 考虑基于动量的权重重要性评估。

8. **多目标剪枝优化**
   提出一个同时优化模型精度、推理延迟和能耗的剪枝框架。定义合适的多目标优化问题，并讨论如何找到Pareto最优解。
   
   *Hint*: 考虑使用加权和方法或ε-约束方法处理多目标。

<details>
<summary>答案</summary>

1. 稀疏度 = 6/16 = 0.375 = 37.5%

2. 剪枝后参数量 = 32 × 128 × 3 × 3 = 36,864，是原参数量 73,728 的 50%

3. $s_{5500} = 0.9 + (0 - 0.9)(1 - \frac{5500-1000}{10000-1000})^3 = 0.9 - 0.9 \times 0.5^3 = 0.7875$

4. 保留 $[-0.5, 0.3]$，剪枝 $[0.1, -0.2]$，剩余L2范数 = $\sqrt{0.25 + 0.09} = 0.583$

5. 对于标准正态分布，$P(|W| < \theta) = 2\Phi(\theta) - 1 = s$，解得 $\theta = \Phi^{-1}((1+s)/2)$，对于 $\mathcal{N}(0, \sigma^2)$，阈值为 $\sigma\theta$

6. 使用对角近似时的误差界：$|\Delta\mathcal{L}_{\text{true}} - \Delta\mathcal{L}_{\text{diag}}| \leq \frac{1}{2}w_i^2 \sum_{j \neq i} |H_{ij}|$

7. 一种策略：基于指数移动平均的重要性 $I_t = \beta I_{t-1} + (1-\beta)|\nabla w|$，只允许重要性低于动态阈值的权重被剪枝

8. 多目标函数：$\min_{\mathbf{M}} [\mathcal{L}_{\text{acc}}(\mathbf{M}), T_{\text{latency}}(\mathbf{M}), E_{\text{energy}}(\mathbf{M})]$，使用NSGA-II等算法寻找Pareto前沿

</details>