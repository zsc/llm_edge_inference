# 第6章：旋转量化与极低比特量化

在前面的章节中，我们探讨了GPTQ、AWQ等主流量化方法，这些方法通常将模型量化到INT8或INT4精度。然而，随着边缘设备对模型大小和计算效率的要求越来越苛刻，研究者们开始探索更激进的量化方案：如何将模型压缩到INT2甚至二值/三值精度，同时保持可接受的性能？本章将深入探讨旋转量化（QuaRot）这一革命性技术，以及极低比特量化的最新进展。

## 6.1 QuaRot：旋转量化的数学原理

QuaRot（Quaternion Rotation Quantization）是一种通过旋转变换来改善量化误差分布的创新方法。其核心思想是：通过适当的正交变换，可以使激活值和权重的分布更加均匀，从而减少量化误差。

### 6.1.1 旋转不变性与量化误差

考虑一个线性层的计算：
$$y = Wx + b$$

其中 $W \in \mathbb{R}^{m \times n}$ 是权重矩阵，$x \in \mathbb{R}^n$ 是输入激活值。

传统量化直接对 $W$ 和 $x$ 进行量化：
$$y_{quant} = Q(W) \cdot Q(x) + b$$

QuaRot 的核心洞察是引入正交矩阵 $R \in \mathbb{R}^{n \times n}$（满足 $R^T R = I$）：
$$y = W(RR^T)x + b = (WR)(R^T x) + b$$

定义变换后的权重和激活值：
- $\tilde{W} = WR$
- $\tilde{x} = R^T x$

则有：
$$y = \tilde{W} \tilde{x} + b$$

关键观察：虽然计算结果不变，但 $\tilde{W}$ 和 $\tilde{x}$ 的分布可能比原始的 $W$ 和 $x$ 更适合量化。

**量化误差分析**

设量化函数为 $Q(\cdot)$，量化误差为：
$$\epsilon_W = Q(W) - W, \quad \epsilon_x = Q(x) - x$$

传统量化的误差：
$$\epsilon_{direct} = Q(W)Q(x) - Wx = W\epsilon_x + \epsilon_W x + \epsilon_W \epsilon_x$$

旋转量化的误差：
$$\epsilon_{rot} = Q(\tilde{W})Q(\tilde{x}) - \tilde{W}\tilde{x} = \tilde{W}\epsilon_{\tilde{x}} + \epsilon_{\tilde{W}} \tilde{x} + \epsilon_{\tilde{W}} \epsilon_{\tilde{x}}$$

通过选择合适的旋转矩阵 $R$，可以使 $\|\epsilon_{\tilde{W}}\|$ 和 $\|\epsilon_{\tilde{x}}\|$ 更小。

### 6.1.2 Hadamard变换与随机旋转

QuaRot 中常用的旋转矩阵包括：

**1. Hadamard矩阵**

Hadamard矩阵 $H_n$ 是一个正交矩阵，其元素只有 $\pm 1/\sqrt{n}$：
$$H_2 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

递归构造：
$$H_{2n} = \frac{1}{\sqrt{2}}\begin{pmatrix} H_n & H_n \\ H_n & -H_n \end{pmatrix}$$

Hadamard变换的优势：
- 计算复杂度仅为 $O(n \log n)$（通过快速Hadamard变换）
- 能有效地将稀疏激活值分散到所有维度

**2. 随机正交矩阵**

生成随机正交矩阵的方法：
1. 生成随机高斯矩阵 $A \in \mathbb{R}^{n \times n}$
2. 进行QR分解：$A = QR$
3. $Q$ 即为随机正交矩阵

随机旋转的理论保证：根据Johnson-Lindenstrauss引理的推广，随机投影能以高概率保持向量间的内积：
$$\mathbb{P}[|(Rx)^T(Ry) - x^T y| > \epsilon \|x\|\|y\|] < 2\exp(-cn\epsilon^2)$$

### 6.1.3 激活值分布的均匀化

LLM中的激活值通常具有以下特征：
1. **异常值（outliers）**：某些维度的值远大于其他维度
2. **稀疏性**：许多维度接近零
3. **非均匀分布**：不同通道的值域差异很大

**旋转后的分布改善**

设原始激活值 $x$ 的各维度方差为 $\{\sigma_i^2\}_{i=1}^n$，经过正交变换 $\tilde{x} = R^T x$ 后：

对于Hadamard变换，有：
$$\text{Var}[\tilde{x}_i] = \frac{1}{n}\sum_{j=1}^n \sigma_j^2$$

即所有维度的方差趋于平均，这种均匀化效应极大地改善了量化性能。

**异常值处理**

设 $x$ 中第 $k$ 维是异常值，$|x_k| \gg |x_i|, i \neq k$。经过Hadamard变换后：
$$\tilde{x}_i = \frac{1}{\sqrt{n}}\sum_{j=1}^n H_{ij} x_j$$

异常值 $x_k$ 的影响被分散到所有 $n$ 个维度，每个维度只承担 $x_k/\sqrt{n}$ 的贡献。

### 6.1.4 计算复杂度分析

**存储开销**
- 需要存储旋转矩阵 $R$：如果使用Hadamard矩阵，由于其结构化特性，无需显式存储
- 如果使用随机矩阵，需要 $O(n^2)$ 存储

**计算开销**

传统矩阵乘法：$y = Wx$
- 计算复杂度：$O(mn)$

旋转量化：$y = (WR)(R^T x)$
1. 计算 $\tilde{x} = R^T x$：
   - Hadamard变换：$O(n \log n)$
   - 一般正交变换：$O(n^2)$
2. 计算 $y = \tilde{W}\tilde{x}$：$O(mn)$

总复杂度：
- 使用Hadamard：$O(mn + n \log n)$
- 使用一般正交矩阵：$O(mn + n^2)$

对于大型模型（$m, n \gg 1$），额外开销相对较小。

**优化技巧**

1. **块对角旋转**：将大矩阵分块，每块使用独立的小旋转矩阵，减少计算开销
2. **层间共享**：多个层共享同一旋转矩阵
3. **融合计算**：将旋转操作与其他操作（如LayerNorm）融合

## 6.2 INT4/INT2/三值网络

随着量化比特数的降低，量化误差呈指数级增长。本节探讨如何在极低比特精度下保持模型性能。

### 6.2.1 极低比特量化的理论基础

**信息论视角**

从信息论角度，n-bit量化最多能表示 $2^n$ 个不同的值。量化过程可以视为一种有损压缩：

设原始权重 $w$ 的熵为 $H(w)$，量化后权重 $\hat{w}$ 的熵最多为 $n$ bits。信息损失为：
$$I_{loss} = H(w) - H(\hat{w}) \geq H(w) - n$$

对于极低比特量化（$n \leq 4$），信息损失巨大，需要特殊技术来补偿。

**量化误差的统计特性**

假设权重 $w \sim \mathcal{N}(0, \sigma^2)$，使用均匀量化器，量化步长为 $\Delta$：

1. **量化噪声功率**：
   $$\sigma_q^2 = \frac{\Delta^2}{12}$$

2. **信噪比（SNR）**：
   $$\text{SNR} = 10\log_{10}\left(\frac{\sigma^2}{\sigma_q^2}\right) \approx 6.02n + 4.77 \text{ dB}$$

   其中 $n$ 是量化比特数。

3. **极低比特下的SNR**：
   - INT4: ~29 dB
   - INT2: ~17 dB
   - 二值: ~11 dB

### 6.2.2 INT4量化：平衡点与实践

INT4量化在精度和压缩率之间达到了良好的平衡，是当前边缘部署的主流选择。

**对称vs非对称量化**

1. **对称量化**：
   $$Q(w) = \text{round}\left(\frac{w}{s}\right) \cdot s$$
   
   其中 $s$ 是缩放因子，量化范围为 $[-7s, 7s]$（对于signed INT4）。

2. **非对称量化**：
   $$Q(w) = \text{round}\left(\frac{w - z}{s}\right) \cdot s + z$$
   
   其中 $z$ 是零点偏移，可以更好地处理非对称分布。

**最优量化参数选择**

给定权重分布，最优缩放因子 $s^*$ 通过最小化量化误差获得：
$$s^* = \arg\min_s \mathbb{E}[(w - Q(w, s))^2]$$

对于正态分布 $w \sim \mathcal{N}(0, \sigma^2)$，INT4对称量化的最优缩放因子约为：
$$s^* \approx 2.5\sigma$$

**群组量化（Group Quantization）**

将权重矩阵分组，每组使用独立的量化参数：
$$W = [W_1, W_2, ..., W_g]$$

每组 $W_i$ 使用独立的 $(s_i, z_i)$。典型配置：
- 组大小：128或256（与硬件向量化单元对齐）
- 存储开销：每组需要额外存储 $s_i$（FP16）和 $z_i$（INT8）

### 6.2.3 INT2与二值化网络

**INT2量化**

INT2只能表示4个值，例如：$\{-3, -1, 1, 3\}$。关键技术：

1. **非均匀量化**：
   使用K-means聚类找到最优的4个代表值：
   $$\{c_1, c_2, c_3, c_4\} = \arg\min_{c} \sum_i \min_j |w_i - c_j|^2$$

2. **学习型量化**：
   将量化阈值作为可学习参数：
   $$Q(w) = \begin{cases}
   c_1, & w < t_1 \\
   c_2, & t_1 \leq w < t_2 \\
   c_3, & t_2 \leq w < t_3 \\
   c_4, & w \geq t_3
   \end{cases}$$

**二值化网络（BNN）**

二值化将权重限制为 $\{-1, +1\}$：
$$Q(w) = \text{sign}(w) = \begin{cases}
+1, & w \geq 0 \\
-1, & w < 0
\end{cases}$$

**梯度估计**：由于sign函数不可微，需要使用直通估计器（STE）：
$$\frac{\partial Q(w)}{\partial w} \approx \begin{cases}
1, & |w| \leq 1 \\
0, & \text{otherwise}
\end{cases}$$

**XNOR-Net优化**：
二值网络的卷积可以用XNOR和popcount操作实现：
$$y = \text{popcount}(\text{XNOR}(Q(W), Q(x))) \cdot \alpha$$

其中 $\alpha$ 是缩放因子，计算复杂度大幅降低。

### 6.2.4 三值网络：{-1, 0, +1}量化

三值网络（Ternary Weight Networks, TWN）在二值网络基础上增加了零值，提供了更好的表达能力。

**量化函数**：
$$Q(w) = \begin{cases}
+1, & w > \Delta \\
0, & |w| \leq \Delta \\
-1, & w < -\Delta
\end{cases}$$

**阈值优化**：
最优阈值 $\Delta^*$ 通过最小化量化误差获得：
$$\Delta^* = \arg\min_\Delta \sum_i |w_i - Q(w_i, \Delta)|^2$$

对于正态分布的权重，最优阈值约为：
$$\Delta^* \approx 0.7\sigma$$

约有60%的权重被量化为零，带来额外的稀疏性优势。

**计算优化**：
三值乘法可以简化为：
- 当权重为+1：直接使用激活值
- 当权重为0：结果为0（跳过计算）
- 当权重为-1：激活值取负

这使得三值网络在某些硬件上比二值网络更高效。

**混合精度三值化**：
关键层（如第一层和最后一层）保持较高精度：
$$W_{mixed} = \begin{cases}
W_{fp16}, & \text{layer} \in \{1, L\} \\
Q_{ternary}(W), & \text{otherwise}
\end{cases}$$

## 6.3 混合精度量化策略

在实际部署中，统一使用单一精度往往不是最优选择。混合精度量化通过为不同层或不同操作分配不同的量化精度，在模型大小和精度之间达到更好的平衡。

### 6.3.1 层级敏感度分析

不同层对量化的敏感度差异很大，需要系统的分析方法来指导精度分配。

**基于Hessian的敏感度度量**

考虑损失函数 $\mathcal{L}$ 关于第 $l$ 层权重 $W^{(l)}$ 的二阶泰勒展开：
$$\Delta\mathcal{L} \approx \text{tr}(\Delta W^{(l)T} \nabla_{W^{(l)}}\mathcal{L}) + \frac{1}{2}\text{tr}(\Delta W^{(l)T} H^{(l)} \Delta W^{(l)})$$

其中 $H^{(l)}$ 是Hessian矩阵。由于一阶项在最优点处为零，量化误差主要由二阶项决定：
$$\Delta\mathcal{L}_q \approx \frac{1}{2}\text{tr}(\epsilon^{(l)T} H^{(l)} \epsilon^{(l)})$$

其中 $\epsilon^{(l)} = Q(W^{(l)}) - W^{(l)}$ 是量化误差。

**实用的敏感度指标**

1. **平均Hessian迹**：
   $$S_H^{(l)} = \frac{1}{n_l}\text{tr}(H^{(l)})$$
   
   其中 $n_l$ 是第 $l$ 层的参数数量。

2. **Fisher信息近似**：
   $$S_F^{(l)} = \mathbb{E}_{x \sim D}\left[\|\nabla_{W^{(l)}}\mathcal{L}(x)\|_F^2\right]$$

3. **激活值范围**：
   $$S_R^{(l)} = \mathbb{E}_{x \sim D}\left[\max_i |a_i^{(l)}| - \min_i |a_i^{(l)}|\right]$$
   
   其中 $a^{(l)}$ 是第 $l$ 层的激活值。

**典型的敏感度模式**

在LLM中，通常观察到以下模式：
1. **嵌入层**：极高敏感度，建议保持FP16或INT8
2. **注意力投影层**：中等敏感度，可使用INT4
3. **FFN层**：较低敏感度，可使用INT2或三值
4. **输出层**：高敏感度，建议保持高精度

### 6.3.2 动态精度分配

**优化问题形式化**

给定总比特预算 $B$，寻找最优的精度分配：
$$\min_{\{b_l\}} \mathcal{L}_{quant} \quad \text{s.t.} \quad \sum_{l=1}^L n_l \cdot b_l \leq B$$

其中 $b_l \in \{1, 2, 4, 8\}$ 是第 $l$ 层的量化比特数。

**贪心算法**

1. 初始化所有层为最低精度
2. 迭代地选择提升精度收益最大的层：
   $$l^* = \arg\max_l \frac{\Delta\mathcal{L}_l}{\Delta B_l}$$
   
   其中 $\Delta\mathcal{L}_l$ 是提升第 $l$ 层精度带来的损失减少，$\Delta B_l$ 是额外的比特开销。

**强化学习方法**

将精度分配建模为马尔可夫决策过程（MDP）：
- **状态**：当前的精度分配 $\{b_l\}$
- **动作**：选择一个层并改变其精度
- **奖励**：$-\mathcal{L}_{quant} - \lambda \cdot \text{BitUsage}$

使用策略梯度方法（如PPO）学习最优策略。

### 6.3.3 关键层保护策略

某些层对模型性能至关重要，需要特殊保护。

**关键层识别**

1. **梯度流分析**：
   梯度范数较大的层通常更重要：
   $$I_g^{(l)} = \mathbb{E}_{x \sim D}\left[\|\frac{\partial\mathcal{L}}{\partial W^{(l)}}\|_F\right]$$

2. **注意力模式分析**：
   对于Transformer模型，分析注意力矩阵的秩：
   $$I_a^{(l)} = \frac{1}{h}\sum_{i=1}^h \text{rank}(A_i^{(l)})$$
   
   其中 $h$ 是注意力头数，$A_i^{(l)}$ 是第 $i$ 个头的注意力矩阵。

**保护策略**

1. **最小精度保证**：
   $$b_l \geq b_{min}^{(l)} = \begin{cases}
   8, & l \in \text{CriticalLayers} \\
   4, & l \in \text{ImportantLayers} \\
   2, & \text{otherwise}
   \end{cases}$$

2. **渐进式量化**：
   先量化非关键层，观察性能影响后再决定是否量化关键层。

3. **补偿机制**：
   为关键层添加可学习的缩放因子和偏置：
   $$y = \gamma \cdot Q(Wx) + \beta$$
   
   其中 $\gamma, \beta$ 通过少量数据微调获得。

### 6.3.4 硬件友好的混合精度设计

实际部署时，混合精度设计必须考虑硬件约束。

**内存对齐要求**

多数硬件要求数据按特定边界对齐：
- INT2: 4个权重打包成一个字节
- INT4: 2个权重打包成一个字节

设计原则：同一层内使用统一精度，避免复杂的打包/解包操作。

**计算单元利用率**

现代硬件通常有专门的低精度计算单元：
- NVIDIA Tensor Core: INT8/INT4 GEMM
- ARM SVE2: INT8点积指令
- Qualcomm Hexagon: HVX向量单元

精度分配应考虑硬件能力：
$$\text{Efficiency}(b_l) = \frac{\text{PeakOps}(b_l)}{\text{PeakOps}(FP32)}$$

**流水线设计**

混合精度推理的流水线优化：
1. **精度分组**：将相同精度的层分组，减少精度切换开销
2. **异步执行**：高精度层和低精度层并行执行
3. **动态调度**：根据层的计算密度动态分配计算资源

**功耗优化**

不同精度的功耗差异显著：
$$P(b) \propto b^2 \cdot f$$

其中 $f$ 是操作频率。混合精度策略：
- 计算密集层：使用低精度降低功耗
- 内存密集层：精度影响较小，可保持较高精度
