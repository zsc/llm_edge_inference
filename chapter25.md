# 第25章：神经架构搜索（NAS）

神经架构搜索（NAS）技术的出现，使得自动设计高效的神经网络架构成为可能。在边缘推理场景中，NAS不再是单纯追求模型精度的工具，而是需要在精度、延迟、能耗、内存占用等多个维度进行平衡的复杂优化问题。本章将深入探讨如何将NAS技术应用于边缘侧模型设计，包括搜索空间的构建、硬件感知的优化策略、多目标优化方法，以及如何将NAS与其他压缩技术相结合，形成完整的自动化模型优化流程。通过本章学习，读者将掌握设计边缘友好架构的系统性方法，并理解如何根据具体硬件平台和应用需求定制化搜索策略。

## 25.1 边缘导向的NAS

传统的NAS方法主要关注在大规模数据集上获得最高的分类精度，而边缘导向的NAS需要从根本上重新思考搜索空间和优化目标。边缘设备的资源限制要求我们在设计搜索空间时就考虑硬件友好性，而不是事后进行压缩。

### 25.1.1 搜索空间设计原则

边缘友好的搜索空间设计需要遵循以下原则：

**1. 操作选择的硬件亲和性**

搜索空间中的基本操作应该在目标硬件上有高效实现。例如，深度可分离卷积（Depthwise Separable Convolution）在移动设备上的效率远高于标准卷积：

标准卷积的计算量：$O(H \times W \times C_{in} \times C_{out} \times K^2)$

深度可分离卷积的计算量：$O(H \times W \times C_{in} \times K^2 + H \times W \times C_{in} \times C_{out})$

计算量减少比例：$\frac{1}{C_{out}} + \frac{1}{K^2}$

**2. 激活函数的选择**

不同激活函数在边缘设备上的性能差异显著。例如，ReLU6相比标准ReLU在量化时更稳定：

$$\text{ReLU6}(x) = \min(\max(0, x), 6)$$

这种有界激活函数避免了激活值的无限增长，有利于定点量化。

**3. 层级连接模式**

搜索空间应包含各种高效的连接模式：
- 残差连接：有助于训练深层网络
- 密集连接：提高特征复用，但增加内存访问
- 线性瓶颈结构：减少计算量的同时保持表达能力

### 25.1.2 边缘友好的操作集合

典型的边缘NAS搜索空间包含以下操作：

**1. 卷积变体**
- 3×3深度可分离卷积
- 5×5深度可分离卷积（通过两个3×3实现）
- 1×1卷积（通道混合）
- 组卷积（groups > 1）

**2. 池化操作**
- 3×3平均池化
- 3×3最大池化
- 自适应平均池化

**3. 特殊结构**
- Inverted Residual Block (MobileNetV2)
- Squeeze-and-Excitation模块（轻量版本）
- Ghost模块（特征图复用）

每种操作的相对成本可以通过以下公式估算：

$$\text{Cost} = \alpha \cdot \text{FLOPs} + \beta \cdot \text{Memory Access} + \gamma \cdot \text{Latency}$$

其中$\alpha, \beta, \gamma$是根据具体硬件平台调整的权重系数。

### 25.1.3 搜索策略对比

**1. DARTS (Differentiable Architecture Search)**

DARTS通过连续松弛将离散的架构搜索问题转化为可微分优化：

$$\bar{o}^{(i,j)} = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o' \in \mathcal{O}} \exp(\alpha_{o'}^{(i,j)})} \cdot o(x^{(i)})$$

其中$\alpha_o^{(i,j)}$是操作$o$在边$(i,j)$上的架构参数。

优化目标采用双层优化：
$$\min_\alpha \mathcal{L}_{val}(w^*(\alpha), \alpha)$$
$$\text{s.t. } w^*(\alpha) = \argmin_w \mathcal{L}_{train}(w, \alpha)$$

**2. ENAS (Efficient Neural Architecture Search)**

ENAS通过参数共享大幅减少搜索成本。控制器使用LSTM生成架构决策：

$$P(\mathcal{A}) = \prod_{t=1}^T P(a_t | a_{1:t-1}; \theta_c)$$

其中$\theta_c$是控制器参数，通过REINFORCE算法更新：

$$\nabla_{\theta_c} J = \mathbb{E}_{\mathcal{A} \sim P(\mathcal{A}; \theta_c)}[(R(\mathcal{A}) - b) \nabla_{\theta_c} \log P(\mathcal{A}; \theta_c)]$$

**3. ProxylessNAS**

ProxylessNAS直接在目标硬件上进行搜索，避免了代理任务的误差：

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda_1 \cdot \text{Latency} + \lambda_2 \cdot \text{Params}$$

延迟预测通过查找表实现：
$$\text{Latency} = \sum_{l} \sum_{o \in \mathcal{O}} p_o^{(l)} \cdot \text{lat}_o^{(l)}$$

### 25.1.4 案例分析：MobileNetV3的NAS过程

MobileNetV3的设计展示了NAS在实际产品中的应用：

**1. 搜索空间定义**
- 基础块：MobileNetV2的倒残差结构
- 扩展因子：{3, 4, 6}
- 卷积核大小：{3, 5, 7}
- SE模块：{有, 无}
- 激活函数：{ReLU, h-swish}

**2. 多目标优化**

目标函数结合了精度和延迟：
$$\text{Reward} = \text{ACC}(m) \times [\frac{\text{LAT}(m)}{\text{TAR}}]^w$$

其中：
- ACC(m)是模型m的精度
- LAT(m)是实测延迟
- TAR是目标延迟
- w是控制精度-延迟权衡的指数（典型值-0.07）

**3. 平台特定优化**

MobileNetV3针对不同平台进行了定制：
- ARM CPU：减少内存重排，优化缓存使用
- GPU：增加并行度，使用更规则的操作
- NPU：选择硬件加速的操作子集

通过NAS找到的架构在ImageNet上达到75.2%的top-1精度，同时在Pixel手机上的延迟仅为66ms。

## 25.2 硬件感知搜索空间

硬件感知的NAS不仅要考虑理论计算量（FLOPs），更要关注实际硬件上的执行效率。不同硬件平台有着截然不同的特性：CPU注重缓存友好性，GPU偏好高并行度操作，而专用加速器则有固定的操作模式。本节探讨如何将这些硬件特性融入搜索空间设计。

### 25.2.1 硬件建模与延迟预测

准确的硬件建模是硬件感知NAS的基础。常用的建模方法包括：

**1. 查找表方法（Lookup Table）**

最直接的方法是为每个操作在目标硬件上实测延迟：

$$\text{Latency}_{total} = \sum_{i=1}^{L} \text{LUT}[\text{op}_i, \text{config}_i]$$

其中LUT存储了操作类型和配置（输入尺寸、通道数等）到延迟的映射。

查找表的构建过程：
- 枚举所有可能的操作配置
- 在目标硬件上运行基准测试
- 考虑不同batch size的影响
- 处理缓存预热和功耗稳定性

**2. 分析模型（Analytical Model）**

基于硬件特性构建延迟预测模型：

$$T_{op} = \max(T_{compute}, T_{memory})$$

计算时间：
$$T_{compute} = \frac{\text{FLOPs}}{f \times n_{cores} \times \text{utilization}}$$

内存访问时间：
$$T_{memory} = \frac{\text{Memory Access}}{\text{Bandwidth} \times \text{efficiency}}$$

**3. 机器学习预测器**

使用神经网络学习从操作特征到延迟的映射：

$$\hat{t} = f_{NN}([\text{op\_type}, H, W, C_{in}, C_{out}, K, S, \text{groups}])$$

训练数据通过随机采样架构并实测获得。预测器的准确性直接影响搜索质量。

### 25.2.2 搜索空间的硬件约束

**1. 内存层次感知**

边缘设备的内存层次对性能影响巨大：

```
寄存器 < L1缓存 < L2缓存 < 主内存 < 外部存储
```

搜索空间设计需要考虑：
- 特征图大小是否适配缓存
- 权重复用模式
- 数据访问局部性

例如，深度可分离卷积的内存访问模式：
- Depthwise阶段：$O(H \times W \times C \times K^2)$
- Pointwise阶段：$O(H \times W \times C_{in} \times C_{out})$

**2. 并行度约束**

不同硬件的并行能力差异：

ARM CPU（NEON）：
- SIMD宽度：128位（4个float32）
- 优化策略：向量化友好的通道数（4的倍数）

GPU（例如Mali）：
- Warp大小：16或32线程
- 优化策略：规则的张量形状，避免分支

DSP（Hexagon）：
- HVX向量单元：1024位宽
- 优化策略：大批量处理，减少控制开销

**3. 量化友好性**

搜索空间应考虑量化后的性能：

$$\text{Quant\_Error} = \frac{||W - Q(W)||_2}{||W||_2}$$

量化友好的设计原则：
- 避免极端的激活值分布
- 使用对称的权重分布
- 限制通道扩展比例

### 25.2.3 跨平台搜索策略

**1. 多平台联合优化**

同时优化多个硬件平台的性能：

$$\min_{\alpha} \sum_{p \in \mathcal{P}} w_p \cdot \text{Latency}_p(\alpha)$$

其中$\mathcal{P}$是目标平台集合，$w_p$是平台权重。

**2. 平台特定分支**

使用条件执行适配不同平台：

```
if platform == "CPU":
    block = DepthwiseSeparable(expand_ratio=3)
elif platform == "GPU":
    block = RegularConv(groups=4)
else:  # NPU
    block = SpecializedBlock()
```

**3. 迁移学习策略**

从一个平台的搜索结果迁移到另一个平台：

$$\alpha_{new} = \alpha_{base} + \Delta\alpha$$

其中$\Delta\alpha$通过少量平台特定搜索获得。

### 25.2.4 能耗感知的架构搜索

能耗是边缘设备的关键约束，需要在搜索过程中显式建模：

**1. 能耗模型**

总能耗包含动态和静态部分：

$$E_{total} = E_{dynamic} + E_{static}$$

动态能耗：
$$E_{dynamic} = \sum_{op} (C_{op} \times V^2 \times f \times \text{Activity})$$

其中：
- $C_{op}$：操作的等效电容
- $V$：工作电压
- $f$：时钟频率
- Activity：操作活跃度

**2. 能耗-性能权衡**

多目标优化中的能耗考虑：

$$\text{EDP} = E \times T$$  （能量延迟积）

或使用更复杂的度量：

$$\text{Metric} = \frac{\text{Accuracy}^\alpha}{\text{Energy}^\beta \times \text{Latency}^\gamma}$$

**3. 动态电压频率调节（DVFS）**

搜索时考虑DVFS的影响：

$$f_{opt} = \argmin_{f} E(f) \times T(f)$$

满足约束：$T(f) \leq T_{deadline}$

**4. 实际案例：能耗优化的轻量级架构**

MCUNet专门为微控制器设计的架构搜索：
- 内存限制：256KB SRAM
- 计算限制：<10 MFLOPS
- 能耗目标：<1mJ per inference

通过联合优化架构和执行调度，在ImageNet的子集上达到70.7%精度，同时满足严格的资源约束。

## 25.3 多目标优化策略

边缘部署的神经架构搜索本质上是一个多目标优化问题，需要在精度、延迟、能耗、内存占用等多个相互冲突的目标之间寻找平衡。本节详细探讨如何设计和求解这类多目标优化问题。

### 25.3.1 Pareto前沿与权衡分析

**1. Pareto最优性定义**

一个解$x$支配另一个解$y$（记作$x \prec y$），当且仅当：
- 对所有目标$i$：$f_i(x) \leq f_i(y)$
- 至少存在一个目标$j$：$f_j(x) < f_j(y)$

Pareto前沿是所有非支配解的集合：
$$\mathcal{P} = \{x \in \mathcal{X} | \nexists y \in \mathcal{X}, y \prec x\}$$

**2. 多目标问题形式化**

边缘NAS的典型多目标优化形式：

$$\min_{\alpha} \mathbf{f}(\alpha) = [f_1(\alpha), f_2(\alpha), ..., f_k(\alpha)]^T$$

其中：
- $f_1(\alpha) = -\text{Accuracy}(\alpha)$ （最小化负精度）
- $f_2(\alpha) = \text{Latency}(\alpha)$
- $f_3(\alpha) = \text{Energy}(\alpha)$
- $f_4(\alpha) = \text{Memory}(\alpha)$

**3. 权衡分析方法**

超体积（Hypervolume）指标：
$$HV(\mathcal{S}) = \lambda(\bigcup_{x \in \mathcal{S}} [f(x), r])$$

其中$\lambda$是Lebesgue测度，$r$是参考点。

归一化权衡度量：
$$\text{Trade-off} = \frac{\Delta f_1 / f_1}{\Delta f_2 / f_2}$$

表示目标1相对变化1%时，目标2的相对变化。

### 25.3.2 演化算法vs梯度方法

**1. NSGA-II（非支配排序遗传算法）**

适用于离散搜索空间的经典方法：

非支配排序：
- 找出所有非支配解，标记为第1层
- 从剩余解中找出非支配解，标记为第2层
- 重复直到所有解被分层

拥挤度距离计算：
$$CD_i = \sum_{m=1}^{M} \frac{f_m^{i+1} - f_m^{i-1}}{f_m^{max} - f_m^{min}}$$

选择策略：优先选择非支配等级低的，同等级内选择拥挤度大的。

**2. 梯度基础的多目标优化**

多梯度下降算法（MGDA）：

寻找共同下降方向：
$$\min_{d} \max_{i} \langle \nabla f_i(\alpha), d \rangle$$

等价于找到梯度凸包中距离原点最近的点：
$$d^* = \argmin_{d \in conv\{\nabla f_1, ..., \nabla f_k\}} ||d||^2$$

**3. 混合方法：GDAS-NSGA**

结合梯度搜索的效率和演化算法的全局搜索能力：

```
1. 使用GDAS快速找到高质量架构候选
2. 将候选作为NSGA-II的初始种群
3. 通过演化探索Pareto前沿
4. 周期性地用梯度方法细化解
```

### 25.3.3 约束优化技术

**1. 硬约束处理**

边缘设备的硬性限制（如内存上限）：

$$\begin{aligned}
\min_{\alpha} & \quad \mathbf{f}(\alpha) \\
\text{s.t.} & \quad g_j(\alpha) \leq 0, \quad j = 1, ..., m
\end{aligned}$$

罚函数方法：
$$\tilde{f}_i(\alpha) = f_i(\alpha) + \lambda \sum_{j} \max(0, g_j(\alpha))^2$$

**2. 软约束与目标转换**

将约束转化为额外目标：

原问题：$\min f(\alpha)$ s.t. $g(\alpha) \leq \epsilon$

转换为：$\min [f(\alpha), g(\alpha)]$

**3. 渐进式约束收紧**

动态调整约束边界：
$$\epsilon_t = \epsilon_{final} + (\epsilon_{init} - \epsilon_{final}) \cdot e^{-t/\tau}$$

使搜索过程从宽松逐渐过渡到严格约束。

### 25.3.4 实际案例：精度-延迟-能耗的三目标优化

**1. 问题设定**

目标：
- 最大化ImageNet精度
- 最小化移动GPU延迟（< 20ms）
- 最小化能耗（< 50mJ）

搜索空间：基于MobileNetV3的超网络

**2. 多目标搜索策略**

采用分层优化方法：

第一阶段：快速筛选
- 使用预测器估计性能
- 随机采样10000个架构
- 保留Pareto前沿上的前100个

第二阶段：精细搜索
- 对每个候选训练超网络权重
- 使用真实硬件测量延迟和能耗
- 应用NSGA-II演化30代

第三阶段：最终选择
- 从Pareto前沿选择满足约束的解
- 完整训练验证最终性能

**3. 结果分析**

典型的Pareto前沿呈现：
- 低延迟区域（<10ms）：精度65-70%，能耗20-30mJ
- 平衡区域（10-15ms）：精度70-73%，能耗30-40mJ
- 高精度区域（15-20ms）：精度73-75%，能耗40-50mJ

权衡关系：
- 精度每提升1%，延迟增加约1.5ms
- 延迟每减少1ms，能耗减少约2mJ
- 精度与能耗呈近似线性关系

**4. 决策支持**

根据应用场景选择合适的架构：
- 实时应用：选择延迟<12ms的架构
- 电池供电：优先考虑能耗<35mJ
- 质量优先：选择精度最高且满足约束的架构

通过可视化Pareto前沿和交互式选择工具，帮助用户做出明智决策。
