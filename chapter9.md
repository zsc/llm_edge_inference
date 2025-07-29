# 第9章：模型剪枝

模型剪枝是一种通过移除神经网络中冗余或不重要的参数来压缩模型的技术。在边缘设备部署场景中，剪枝不仅能够减少模型的存储需求，还能显著降低推理时的计算量和内存占用。本章将深入探讨剪枝的数学原理、实践策略以及在大语言模型中的应用，重点关注如何在保持模型性能的同时最大化压缩率。

## 9.1 结构化剪枝vs非结构化剪枝

### 9.1.1 非结构化剪枝

非结构化剪枝（Unstructured Pruning）是指在权重矩阵中任意位置移除单个权重元素，不考虑特定的结构模式。

**数学表示**：
对于权重矩阵 $\mathbf{W} \in \mathbb{R}^{m \times n}$，非结构化剪枝通过二值掩码矩阵 $\mathbf{M} \in \{0,1\}^{m \times n}$ 实现：

$$\mathbf{W}_{\text{pruned}} = \mathbf{W} \odot \mathbf{M}$$

其中 $\odot$ 表示逐元素乘法（Hadamard积）。

**稀疏度定义**：
稀疏度 $s$ 定义为零元素的比例：

$$s = \frac{\|\mathbf{M}\|_0}{mn} = \frac{\text{非零元素数量}}{\text{总元素数量}}$$

**优势**：
- 理论上可以达到更高的压缩率
- 灵活性高，可以精确控制剪枝粒度
- 保持模型精度的能力更强

**劣势**：
- 需要专门的稀疏矩阵运算库支持
- 在通用硬件（CPU/GPU）上难以获得实际加速
- 存储需要额外的索引信息

### 9.1.2 结构化剪枝

结构化剪枝（Structured Pruning）按照预定义的结构模式移除参数，如整个通道、滤波器或注意力头。

**主要类型**：

1. **通道剪枝（Channel Pruning）**：
   对于卷积层权重 $\mathbf{W} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K \times K}$，通道剪枝移除整个输入或输出通道：
   
   $$\mathbf{W}_{\text{pruned}} \in \mathbb{R}^{C'_{\text{out}} \times C'_{\text{in}} \times K \times K}$$
   
   其中 $C'_{\text{out}} \leq C_{\text{out}}$，$C'_{\text{in}} \leq C_{\text{in}}$

2. **滤波器剪枝（Filter Pruning）**：
   移除整个滤波器（输出通道）：
   
   $$\mathbf{W}_{i,:,:,:} = \mathbf{0}, \quad i \in \mathcal{P}$$
   
   其中 $\mathcal{P}$ 是被剪枝的滤波器索引集合

3. **注意力头剪枝（Attention Head Pruning）**：
   对于多头注意力，移除整个注意力头：
   
   $$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_i)_{i \notin \mathcal{P}} \mathbf{W}^O$$

**硬件友好性分析**：

结构化剪枝后的计算可以直接映射到密集矩阵运算：
- 无需特殊的稀疏运算支持
- 可以直接使用现有的BLAS/cuBLAS库
- 内存访问模式规则，缓存友好

### 9.1.3 半结构化剪枝

近年来，NVIDIA等硬件厂商开始支持半结构化稀疏模式，如2:4稀疏（每4个连续元素中保留2个）。

**2:4稀疏模式**：
对于向量 $\mathbf{v} = [v_1, v_2, v_3, v_4]$，2:4稀疏要求恰好两个元素为零：

$$\|\mathbf{v}\|_0 = 2$$

这种模式在Ampere架构GPU上可以获得理论上2倍的加速。

**数学优化问题**：
寻找最优的2:4稀疏模式可以表述为：

$$\min_{\mathbf{M}} \|\mathbf{W} - \mathbf{W} \odot \mathbf{M}\|_F^2$$
$$\text{s.t. } \sum_{j=4k}^{4k+3} M_{ij} = 2, \quad \forall i,k$$

### 9.1.4 剪枝模式选择策略

选择合适的剪枝模式需要考虑多个因素：

1. **硬件特性**：
   - 通用CPU/GPU：优先选择结构化剪枝
   - 专用加速器（如稀疏张量核心）：可以考虑半结构化
   - DSP/NPU：需要根据具体架构特性选择

2. **压缩率要求**：
   - 高压缩率（>90%稀疏）：非结构化剪枝
   - 中等压缩率（50-90%）：结构化或半结构化
   - 实时推理要求：结构化剪枝优先

3. **精度敏感性**：
   对于Transformer模型，不同组件的剪枝敏感性排序：
   
   $$\text{Embedding} < \text{FFN} < \text{Attention} < \text{LayerNorm}$$

## 9.2 渐进式剪枝策略

### 9.2.1 渐进幅度剪枝

渐进幅度剪枝（Gradual Magnitude Pruning, GMP）是一种在训练过程中逐步增加稀疏度的方法。

**稀疏度调度函数**：
给定初始稀疏度 $s_0$、目标稀疏度 $s_f$、开始步数 $t_0$ 和结束步数 $t_f$，在步数 $t$ 时的稀疏度为：

$$s_t = s_f + (s_0 - s_f) \left(1 - \frac{t - t_0}{t_f - t_0}\right)^3$$

这里使用三次多项式确保平滑过渡。

**剪枝阈值计算**：
在每个剪枝步骤，计算权重幅度的第 $k$ 个百分位数作为阈值：

$$\theta = \text{Percentile}(|\mathbf{W}|, 100 \cdot s_t)$$

然后更新掩码：

$$M_{ij} = \begin{cases}
1, & \text{if } |W_{ij}| \geq \theta \\
0, & \text{otherwise}
\end{cases}$$

### 9.2.2 迭代剪枝与恢复

迭代剪枝通过多轮"剪枝-恢复训练"循环来达到目标稀疏度。

**算法流程**：
1. 初始化：设置剪枝轮数 $N$，每轮剪枝率 $p$
2. 对于 $i = 1, ..., N$：
   - 剪枝：移除当前最小的 $p\%$ 权重
   - 恢复训练：训练 $T$ 个epoch
   - 更新稀疏度：$s_i = 1 - (1-p)^i$

**理论分析**：
假设每轮剪枝后的精度损失为 $\Delta_i$，且通过恢复训练可以恢复 $\alpha \Delta_i$ 的精度（$0 < \alpha < 1$），则总精度损失约为：

$$\Delta_{\text{total}} \approx \sum_{i=1}^N (1-\alpha)^i \Delta_1$$

当 $\alpha$ 接近1时，多轮迭代可以显著减少精度损失。

### 9.2.3 动态稀疏训练

动态稀疏训练（Dynamic Sparse Training, DST）允许在训练过程中改变稀疏模式。

**权重更新策略**：
1. **剪枝步骤**：移除幅度最小的 $k$ 个权重
2. **生长步骤**：根据梯度信息添加 $k$ 个新连接

梯度准则下的生长策略：
$$\text{Score}_{ij} = |\nabla_{W_{ij}} \mathcal{L}| \cdot |\nabla_{a_j} \mathcal{L}|$$

其中 $\nabla_{a_j} \mathcal{L}$ 是对激活的梯度。

**稀疏模式演化**：
定义稀疏模式的Hamming距离：

$$d_H(\mathbf{M}^{(t)}, \mathbf{M}^{(t+1)}) = \frac{1}{mn}\sum_{i,j} |M_{ij}^{(t)} - M_{ij}^{(t+1)}|$$

好的动态稀疏策略应该在训练后期保持较小的 $d_H$。

### 9.2.4 学习率调度协同

剪枝过程中的学习率调度对最终性能至关重要。

**分段学习率策略**：
- 预热阶段（$t < t_0$）：正常学习率 $\eta_0$
- 剪枝阶段（$t_0 \leq t < t_f$）：递减学习率 $\eta_t = \eta_0 \cdot \cos\left(\frac{\pi(t-t_0)}{2(t_f-t_0)}\right)$
- 微调阶段（$t \geq t_f$）：固定小学习率 $\eta_f = 0.1\eta_0$

**自适应学习率调整**：
基于剪枝引起的梯度范数变化调整学习率：

$$\eta_{\text{adjusted}} = \eta \cdot \frac{\|\nabla \mathcal{L}(\mathbf{W})\|_2}{\|\nabla \mathcal{L}(\mathbf{W}_{\text{pruned}})\|_2}$$

## 9.3 基于重要性的剪枝准则

### 9.3.1 一阶重要性度量

**权重幅度准则**：
最简单直观的重要性度量是权重的绝对值：

$$I_{ij}^{\text{magnitude}} = |W_{ij}|$$

**梯度幅度准则**：
考虑权重对损失函数的影响：

$$I_{ij}^{\text{gradient}} = |W_{ij} \cdot \nabla_{W_{ij}} \mathcal{L}|$$

这近似于移除权重 $W_{ij}$ 对损失函数的一阶影响。

**组合准则**：
结合权重幅度和梯度信息：

$$I_{ij}^{\text{combined}} = |W_{ij}|^\alpha \cdot |\nabla_{W_{ij}} \mathcal{L}|^\beta$$

其中 $\alpha, \beta$ 是超参数，通常 $\alpha + \beta = 1$。

### 9.3.2 二阶重要性度量

**Taylor展开分析**：
将权重 $W_{ij}$ 设为0后的损失变化可以用Taylor展开近似：

$$\Delta \mathcal{L}_{ij} = -W_{ij} g_{ij} + \frac{1}{2} W_{ij}^2 H_{ij,ij} + O(W_{ij}^3)$$

其中 $g_{ij} = \nabla_{W_{ij}} \mathcal{L}$，$H_{ij,ij}$ 是Hessian矩阵的对角元素。

**Fisher信息近似**：
使用Fisher信息矩阵近似Hessian：

$$F_{ij,ij} = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \left[ \left(\frac{\partial \log p(\mathbf{y}|\mathbf{x}, \mathbf{W})}{\partial W_{ij}}\right)^2 \right]$$

重要性得分：

$$I_{ij}^{\text{Fisher}} = \frac{1}{2} W_{ij}^2 F_{ij,ij}$$

**最优脑损伤（OBD）准则**：
假设Hessian矩阵是对角的，重要性为：

$$I_{ij}^{\text{OBD}} = \frac{W_{ij}^2}{2[H^{-1}]_{ij,ij}}$$

### 9.3.3 结构化重要性度量

对于结构化剪枝，需要评估整个结构单元的重要性。

**通道重要性**：
对于第 $c$ 个通道，聚合所有相关权重的重要性：

$$I_c^{\text{channel}} = \sum_{i,j,k} I_{c,i,j,k}$$

**基于特征图的度量**：
使用批归一化的缩放因子作为通道重要性指标：

$$I_c^{\text{BN}} = |\gamma_c|$$

其中 $\gamma_c$ 是批归一化层的缩放参数。

**激活统计度量**：
基于特征图的统计信息：

$$I_c^{\text{activation}} = \mathbb{E}_{\mathbf{x}} \left[ \|\mathbf{a}_c(\mathbf{x})\|_2 \right]$$

### 9.3.4 彩票假设与剪枝

彩票假设（Lottery Ticket Hypothesis）指出，随机初始化的密集网络包含一个子网络（"中奖彩票"），当独立训练时可以达到原网络的精度。

**迭代幅度剪枝（IMP）**：
1. 随机初始化网络 $f(\mathbf{x}; \mathbf{W}_0)$
2. 训练网络至收敛，得到 $\mathbf{W}_T$
3. 剪枝 $p\%$ 最小幅度权重，得到掩码 $\mathbf{M}$
4. 将剩余权重重置为初始值：$\mathbf{W} = \mathbf{W}_0 \odot \mathbf{M}$
5. 重复步骤2-4直到达到目标稀疏度

**学习率回退技巧**：
对于大规模网络，需要回退到训练早期的权重：

$$\mathbf{W}_{\text{rewind}} = \mathbf{W}_k \odot \mathbf{M}, \quad k \ll T$$

通常 $k$ 选择为总训练步数的1-10%。

**理论解释**：
彩票假设可以从优化景观的角度理解：
- 好的稀疏子网络对应于损失景观中的"盆地"
- 适当的初始化使得这些子网络更容易通过梯度下降找到

## 9.4 剪枝后的微调技术

### 9.4.1 学习率回退与微调

剪枝后的微调是恢复模型性能的关键步骤。合适的学习率策略可以显著提升剪枝模型的最终精度。

**学习率回退（Learning Rate Rewinding）**：
将学习率重置到训练过程中的某个早期状态：

$$\eta_{\text{rewind}} = \eta(t_{\text{rewind}})$$

其中 $t_{\text{rewind}}$ 通常选择为原始训练的10-20%位置。

**分层学习率策略**：
对于不同稀疏度的层使用不同的学习率：

$$\eta_l = \eta_{\text{base}} \cdot (1 - s_l)^\alpha$$

其中 $s_l$ 是第 $l$ 层的稀疏度，$\alpha > 0$ 是缩放因子。

**梯度掩码技术**：
确保被剪枝的权重在微调过程中保持为零：

$$\mathbf{W}^{(t+1)} = \mathbf{W}^{(t)} - \eta \cdot (\nabla_{\mathbf{W}} \mathcal{L} \odot \mathbf{M})$$

### 9.4.2 知识蒸馏辅助微调

使用原始密集模型作为教师模型，指导剪枝模型的微调。

**蒸馏损失函数**：
$$\mathcal{L}_{\text{distill}} = (1-\lambda) \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{KD}}$$

其中知识蒸馏损失为：

$$\mathcal{L}_{\text{KD}} = \tau^2 \cdot \text{KL}(p_{\text{student}}(\mathbf{y}/\tau | \mathbf{x}) \| p_{\text{teacher}}(\mathbf{y}/\tau | \mathbf{x}))$$

$\tau$ 是温度参数，通常设为3-5。

**特征蒸馏**：
除了输出层的蒸馏，还可以对中间层特征进行蒸馏：

$$\mathcal{L}_{\text{feature}} = \sum_{l \in \mathcal{L}_{\text{distill}}} \|\mathbf{F}_l^{\text{student}} - \mathbf{F}_l^{\text{teacher}}\|_2^2$$

**自适应蒸馏权重**：
根据剪枝程度动态调整蒸馏权重：

$$\lambda_l = \lambda_0 \cdot \exp(k \cdot s_l)$$

其中 $k > 0$ 控制随稀疏度增加的程度。

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