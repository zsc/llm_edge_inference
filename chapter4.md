# 第4章：后训练量化（PTQ）

后训练量化（Post-Training Quantization, PTQ）是边缘部署中最实用的模型压缩技术之一。与量化感知训练（QAT）相比，PTQ无需重新训练即可将浮点模型转换为低比特表示，显著降低了部署成本。本章深入探讨现代PTQ技术的数学原理、算法设计与工程权衡，帮助读者掌握在资源受限环境下高效部署大语言模型的核心技术。

## 章节大纲

### 4.1 GPTQ：最优量化权重量化
- 4.1.1 GPTQ的数学基础：二阶泰勒展开与Hessian近似
- 4.1.2 逐层量化与最优化问题求解
- 4.1.3 OBS（Optimal Brain Surgeon）理论与应用
- 4.1.4 块级量化与计算复杂度分析

### 4.2 AWQ：激活感知权重量化
- 4.2.1 激活分布对量化的影响分析
- 4.2.2 显著权重识别与保护机制
- 4.2.3 Per-channel缩放因子优化
- 4.2.4 AWQ与GPTQ的对比分析

### 4.3 SmoothQuant：平滑激活异常值
- 4.3.1 LLM中的激活异常值现象
- 4.3.2 激活-权重量化难度迁移
- 4.3.3 平滑因子的数学推导
- 4.3.4 INT8量化的实践考虑

### 4.4 量化粒度与硬件适配
- 4.4.1 量化粒度层次：逐层、逐通道、逐组
- 4.4.2 硬件量化支持：INT8/INT4指令集
- 4.4.3 混合精度策略设计
- 4.4.4 量化格式与存储优化

## 4.1 GPTQ：最优量化权重量化

GPTQ（GPT Quantization）是一种基于最优化理论的后训练量化方法，通过求解带约束的优化问题来最小化量化误差。其核心思想源自经典的OBS（Optimal Brain Surgeon）理论，但针对大规模语言模型进行了显著改进。

### 4.1.1 GPTQ的数学基础：二阶泰勒展开与Hessian近似

考虑一个预训练的神经网络层，权重矩阵为 $\mathbf{W} \in \mathbb{R}^{d_{out} \times d_{in}}$，输入为 $\mathbf{X} \in \mathbb{R}^{n \times d_{in}}$。量化的目标是找到量化权重 $\hat{\mathbf{W}}$，使得输出误差最小：

$$\mathcal{L} = \|\mathbf{XW}^T - \mathbf{X}\hat{\mathbf{W}}^T\|_F^2$$

对于权重的微小扰动 $\delta\mathbf{W} = \hat{\mathbf{W}} - \mathbf{W}$，我们可以通过二阶泰勒展开来近似损失函数的变化：

$$\Delta\mathcal{L} \approx \frac{1}{2}\text{tr}(\delta\mathbf{W}^T \mathbf{H} \delta\mathbf{W})$$

其中 $\mathbf{H} = 2\mathbf{X}^T\mathbf{X}$ 是输入的二阶统计量，可以视为Fisher信息矩阵的近似。这个Hessian矩阵捕获了不同权重之间的相关性，对于优化量化误差至关重要。

### 4.1.2 逐层量化与最优化问题求解

GPTQ采用逐层量化策略，对每一层独立求解优化问题。对于第 $l$ 层，优化目标可以表示为：

$$\min_{\hat{\mathbf{W}}^{(l)}} \|\mathbf{X}^{(l)}(\mathbf{W}^{(l)})^T - \mathbf{X}^{(l)}(\hat{\mathbf{W}}^{(l)})^T\|_F^2$$

其中 $\hat{\mathbf{W}}^{(l)} = q(\mathbf{W}^{(l)})$ 表示量化函数。

为了高效求解，GPTQ将问题分解为行级别的子问题。对于权重矩阵的第 $i$ 行 $\mathbf{w}_i$，优化问题变为：

$$\min_{\hat{\mathbf{w}}_i} (\mathbf{w}_i - \hat{\mathbf{w}}_i)^T \mathbf{H}_{[i,i]} (\mathbf{w}_i - \hat{\mathbf{w}}_i)$$

这里 $\mathbf{H}_{[i,i]}$ 是Hessian矩阵对应第 $i$ 行的子矩阵。

### 4.1.3 OBS（Optimal Brain Surgeon）理论与应用

GPTQ的理论基础来自OBS框架，该框架提供了在保持网络性能的同时修剪或量化权重的最优策略。OBS的核心洞察是：当量化某个权重时，可以通过调整其他相关权重来补偿量化误差。

对于权重 $w_{ij}$ 的量化，最优的补偿更新为：

$$\delta\mathbf{w}_{-ij} = -\frac{w_{ij} - \hat{w}_{ij}}{[\mathbf{H}^{-1}]_{jj}} \mathbf{H}^{-1}_{:,j}$$

其中 $\delta\mathbf{w}_{-ij}$ 表示除 $w_{ij}$ 外其他权重的更新，$\mathbf{H}^{-1}_{:,j}$ 是Hessian逆矩阵的第 $j$ 列。

然而，直接计算和存储完整的Hessian逆矩阵对于大规模模型是不可行的。GPTQ通过以下策略解决这个问题：

1. **块级量化**：将权重矩阵划分为大小为 $B \times B$ 的块，每个块内独立计算Hessian
2. **Cholesky分解**：利用Hessian的正定性，通过Cholesky分解高效求逆
3. **贪心顺序**：按照量化误差从小到大的顺序处理权重

### 4.1.4 块级量化与计算复杂度分析

GPTQ的块级量化策略显著降低了计算复杂度。设权重矩阵维度为 $d \times d$，块大小为 $B$：

- **朴素方法**：$O(d^3)$ 的Hessian求逆复杂度
- **块级方法**：$O(\frac{d}{B} \cdot B^3) = O(d \cdot B^2)$ 的总复杂度

典型设置下（$B = 128$），这带来了数个数量级的加速。块级处理的伪算法如下：

```
对于每个大小为B的块:
    1. 计算块内的局部Hessian: H_block = X_block^T @ X_block
    2. 对H_block进行Cholesky分解: L @ L^T = H_block
    3. 对块内每个权重:
        a. 计算量化误差
        b. 使用OBS公式更新其他权重
        c. 更新Cholesky因子（rank-1更新）
```

**量化格式选择**：GPTQ支持多种量化格式：

- **对称量化**：$\hat{w} = s \cdot \text{round}(w/s)$，其中 $s = \frac{\max(|w|)}{2^{b-1}-1}$
- **非对称量化**：$\hat{w} = s \cdot (\text{round}(w/s + z) - z)$，增加零点 $z$ 提供更大灵活性
- **分组量化**：每 $g$ 个权重共享一个缩放因子，在精度和存储间取得平衡

**实践考虑**：

1. **校准数据集**：GPTQ需要少量校准数据（通常128-512个样本）来估计Hessian
2. **层敏感度**：不同层对量化的敏感度差异很大，可以采用混合精度策略
3. **激活量化**：GPTQ主要关注权重量化，激活量化需要额外处理

## 4.2 AWQ：激活感知权重量化

AWQ（Activation-aware Weight Quantization）是一种创新的后训练量化方法，其核心思想是根据激活分布的特征来指导权重量化。与GPTQ的纯优化方法不同，AWQ认识到并非所有权重对模型输出的影响都是均等的——某些权重通道处理的激活值显著大于其他通道，这些"显著通道"对保持模型性能至关重要。

### 4.2.1 激活分布对量化的影响分析

考虑线性层的计算：$\mathbf{y} = \mathbf{x}\mathbf{W}^T$，其中 $\mathbf{x} \in \mathbb{R}^{1 \times d_{in}}$，$\mathbf{W} \in \mathbb{R}^{d_{out} \times d_{in}}$。量化误差可以表示为：

$$\Delta\mathbf{y} = \mathbf{x}(\mathbf{W} - \hat{\mathbf{W}})^T = \mathbf{x}\Delta\mathbf{W}^T$$

输出误差的第 $j$ 个元素为：

$$\Delta y_j = \sum_{i=1}^{d_{in}} x_i \Delta w_{ji}$$

AWQ的关键观察是：当 $|x_i|$ 很大时，对应的权重列 $\mathbf{w}_{:,i}$ 的量化误差会被放大。因此，这些"重要"通道需要更精确的量化。

通过对大量校准数据的统计分析，AWQ发现激活的显著性具有以下特征：

1. **稀疏性**：只有少数通道（通常1-5%）具有显著大的激活值
2. **持久性**：这些显著通道在不同输入样本间保持相对稳定
3. **层间差异**：不同层的激活分布模式差异很大

### 4.2.2 显著权重识别与保护机制

AWQ使用激活幅度来识别显著权重。对于权重矩阵的第 $i$ 列，其重要性得分定义为：

$$s_i = \mathbb{E}_{\mathbf{x} \sim \mathcal{D}}[|x_i|] \cdot \|\mathbf{w}_{:,i}\|_2$$

这个得分同时考虑了激活幅度和权重范数。基于得分，AWQ采用以下保护策略：

1. **通道级缩放**：对于重要通道，引入缩放因子 $s_i > 1$ 来降低相对量化误差
2. **混合精度**：将top-k%的重要通道保持在更高精度（如INT8而非INT4）
3. **自适应量化范围**：为重要通道分配更大的量化范围

数学上，通道缩放可以表示为：

$$\hat{\mathbf{W}}_{:,i} = \frac{1}{s_i} \cdot q(s_i \cdot \mathbf{W}_{:,i})$$

其中 $s_i$ 的选择需要平衡量化误差和硬件效率。

### 4.2.3 Per-channel缩放因子优化

AWQ的核心创新在于自动学习最优的per-channel缩放因子。优化目标是最小化量化后的重构误差：

$$\min_{\{s_i\}} \mathbb{E}_{\mathbf{x}}\left[\left\|\mathbf{x}\mathbf{W}^T - \mathbf{x}\text{diag}(\mathbf{s})^{-1}q(\text{diag}(\mathbf{s})\mathbf{W})^T\right\|^2\right]$$

其中 $\mathbf{s} = [s_1, s_2, ..., s_{d_{in}}]^T$ 是缩放因子向量。

AWQ使用网格搜索来优化缩放因子：

1. **初始化**：$s_i = 1$ 对所有通道
2. **识别重要通道**：根据激活统计选择top-k%通道
3. **网格搜索**：对每个重要通道，在 $s_i \in \{2^{-3}, 2^{-2.5}, ..., 2^{3}\}$ 范围内搜索
4. **贪心优化**：逐通道优化，保持其他通道的缩放因子固定

这种方法虽然不是全局最优，但在实践中表现出色，特别是对于INT4量化。

### 4.2.4 AWQ与GPTQ的对比分析

AWQ和GPTQ代表了两种不同的量化哲学：

**计算复杂度对比**：
- GPTQ：$O(d_{in}^2 \cdot d_{out})$ 用于Hessian计算和求逆
- AWQ：$O(k \cdot d_{in} \cdot n_{search})$ 其中 $k$ 是重要通道数，$n_{search}$ 是搜索次数

**精度-效率权衡**：

| 方法 | INT4困惑度（↓） | 量化时间 | 内存需求 |
|------|----------------|----------|----------|
| GPTQ | 5.67 | 4小时 | 高（Hessian存储） |
| AWQ  | 5.60 | 0.5小时 | 低（仅统计量） |

**适用场景**：
- **GPTQ**：当计算资源充足，追求理论最优解时
- **AWQ**：快速部署场景，特别是需要频繁重新量化时

**硬件友好性**：
AWQ的per-channel缩放可以高效地在现代硬件上实现：
- GPU：利用tensor core的缩放功能
- ARM：NEON指令集的向量操作
- NPU：许多NPU原生支持per-channel量化

**混合策略**：
实践中可以结合两种方法的优势：
1. 使用AWQ快速识别重要通道
2. 对重要通道应用GPTQ的精确优化
3. 对其余通道使用简单的round-to-nearest量化

## 4.3 SmoothQuant：平滑激活异常值

SmoothQuant解决了LLM量化中的一个关键挑战：激活值中的异常值（outliers）。这些异常值使得INT8激活量化极其困难，而SmoothQuant通过巧妙地在激活和权重之间迁移量化难度来解决这个问题。

### 4.3.1 LLM中的激活异常值现象

在大规模语言模型中，激活值呈现出独特的分布特征：

1. **系统性异常值**：某些特定通道的激活值可能比平均值大100倍以上
2. **位置固定性**：这些异常值总是出现在固定的通道位置
3. **层间传播**：异常值模式从浅层传播到深层

数学上，对于激活向量 $\mathbf{x} \in \mathbb{R}^d$，异常值可以定义为：

$$\text{outlier}_i = \begin{cases}
1, & \text{if } |x_i| > \alpha \cdot \text{median}(|\mathbf{x}|) \\
0, & \text{otherwise}
\end{cases}$$

其中 $\alpha$ 通常设置为20-50。研究表明，仅1%的通道可能包含90%以上的激活能量。

这种分布对量化的影响是灾难性的。考虑INT8量化的动态范围：

$$\text{scale} = \frac{\max(|\mathbf{x}|)}{127}$$

当存在极大的异常值时，大部分正常激活值会被量化到很小的整数范围内，导致严重的精度损失。

### 4.3.2 激活-权重量化难度迁移

SmoothQuant的核心创新是认识到：虽然激活难以量化，但权重通常易于量化。因此，可以通过数学变换将量化难度从激活迁移到权重。

对于线性变换 $\mathbf{y} = \mathbf{x}\mathbf{W}^T$，引入对角缩放矩阵 $\mathbf{S} = \text{diag}(s_1, s_2, ..., s_d)$：

$$\mathbf{y} = \mathbf{x}\mathbf{W}^T = (\mathbf{x}\mathbf{S}^{-1})(\mathbf{S}\mathbf{W}^T) = \hat{\mathbf{x}}\hat{\mathbf{W}}^T$$

其中：
- $\hat{\mathbf{x}} = \mathbf{x}\mathbf{S}^{-1}$ 是平滑后的激活
- $\hat{\mathbf{W}} = \mathbf{W}\mathbf{S}^T$ 是缩放后的权重

关键洞察：通过选择合适的 $\mathbf{S}$，可以：
1. 减小激活中的异常值：$\hat{x}_i = x_i / s_i$
2. 相应地放大权重：$\hat{w}_{ij} = w_{ij} \cdot s_j$
3. 保持数学等价性：输出完全不变

### 4.3.3 平滑因子的数学推导

最优平滑因子的选择需要平衡激活和权重的量化难度。SmoothQuant提出了一个简单而有效的公式：

$$s_j = \left(\frac{\max_i |x_{ij}|^\alpha}{\max_i |w_{ij}|^\alpha}\right)^{\frac{1}{2}}$$

其中 $\alpha$ 是平滑强度超参数（通常设为0.5）。

这个公式的推导基于最小化量化误差的上界。定义量化后的总误差为：

$$E = E_{\text{act}} + E_{\text{weight}} = \|\mathbf{x} - q(\mathbf{x})\|^2 + \|\mathbf{W} - q(\mathbf{W})\|^2_F$$

通过拉格朗日乘数法求解约束优化问题：

$$\min_{\mathbf{s}} E(\mathbf{s}) \quad \text{s.t.} \quad \prod_j s_j = 1$$

可以得到上述平滑因子公式。$\alpha$ 参数控制了量化难度的迁移程度：
- $\alpha = 0$：完全迁移到权重（激活最易量化）
- $\alpha = 1$：不进行迁移（保持原始分布）
- $\alpha = 0.5$：平衡迁移（实践中效果最佳）

### 4.3.4 INT8量化的实践考虑

SmoothQuant使得INT8量化在LLM上成为可能，但仍需要careful engineering：

**1. 离线与在线平滑**

- **离线平滑**：预先计算并存储平滑后的权重
  - 优点：推理时无额外计算
  - 缺点：需要修改模型权重

- **在线平滑**：推理时动态应用平滑
  - 优点：保持原始模型不变
  - 缺点：增加推理延迟

**2. 层级别调整**

不同层的激活分布差异很大，需要层特定的处理：

```
对于每一层:
    1. 收集激活统计：max_act = calibrate_activation(layer, data)
    2. 计算平滑因子：s = compute_smoothing_factor(max_act, layer.weight)
    3. 应用平滑：
        - 如果是离线：layer.weight = layer.weight * s
        - 如果是在线：activation = activation / s
```

**3. 静态vs动态量化**

- **静态量化**：使用校准数据预计算量化参数
  - 激活scale: $s_x = \max_{\text{calib}}(|\hat{\mathbf{x}}|) / 127$
  - 权重scale: $s_w = \max(|\hat{\mathbf{W}}|) / 127$

- **动态量化**：每个输入动态计算激活scale
  - 更准确但计算开销更大
  - 适合batch size较大的场景

**4. 硬件优化考虑**

现代硬件对INT8有良好支持，但需要注意：

- **内存对齐**：确保量化张量满足硬件要求（如16字节对齐）
- **融合操作**：将反量化与后续操作融合以减少内存访问
- **混合精度**：关键层（如最后的lm_head）保持FP16

**性能收益**：

| 模型规模 | FP16内存 | INT8内存 | 加速比 |
|---------|----------|----------|--------|
| 7B | 14GB | 7GB | 1.8x |
| 13B | 26GB | 13GB | 1.9x |
| 70B | 140GB | 70GB | 2.1x |

SmoothQuant的成功表明，通过深入理解模型特性并设计相应的数学变换，可以克服看似不可能的量化挑战。