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

**稀疏存储格式**：
非结构化稀疏矩阵通常使用以下格式存储：
- **COO (Coordinate)格式**：存储 (row, col, value) 三元组
- **CSR (Compressed Sparse Row)格式**：使用行指针数组、列索引数组和值数组
- **CSC (Compressed Sparse Column)格式**：类似CSR但按列存储

对于稀疏度为 $s$ 的矩阵，COO格式的存储需求从 $mn$ 降至 $3(1-s)mn$，只有当 $s > 2/3$ 时才有存储优势。

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