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

### 25.1.5 超网络训练策略

**1. 权重共享机制**

超网络（SuperNet）包含所有可能的子网络，通过权重共享加速搜索：

$$\mathcal{W} = \{W^{(i,j)}_o | (i,j) \in \mathcal{E}, o \in \mathcal{O}\}$$

其中$\mathcal{E}$是边集合，$\mathcal{O}$是操作集合。

单路径采样训练：
$$\mathcal{L}_{train} = \mathbb{E}_{\alpha \sim \mathcal{U}(\mathcal{A})}[\mathcal{L}(x, y; W_\alpha)]$$

其中$W_\alpha$是架构$\alpha$对应的权重子集。

**2. 公平性训练（FairNAS）**

不同操作的训练难度不同，需要确保公平性：

期望训练策略：
$$p(o) = \frac{\exp(\lambda \cdot \mathcal{L}_o)}{\sum_{o' \in \mathcal{O}} \exp(\lambda \cdot \mathcal{L}_{o'})}$$

其中$\mathcal{L}_o$是操作$o$的平均损失，$\lambda$控制采样偏好。

**3. 渐进收缩（Progressive Shrinking）**

逐步减少搜索空间，提高训练稳定性：

温度退火：
$$p_t(o) = \frac{\exp(\alpha_o / T_t)}{\sum_{o'} \exp(\alpha_{o'} / T_t)}$$

其中$T_t = T_0 \cdot \exp(-t/\tau)$是退火温度。

**4. 知识蒸馏加速**

使用教师网络指导超网络训练：

$$\mathcal{L} = (1-\lambda)\mathcal{L}_{CE} + \lambda \mathcal{L}_{KD}$$

其中：
$$\mathcal{L}_{KD} = \tau^2 \cdot KL(p_{student} || p_{teacher})$$

### 25.1.6 早期停止与代理任务

**1. 代理数据集选择**

使用小规模数据集加速搜索：

相关性度量：
$$\rho = \frac{\text{Cov}(R_{proxy}, R_{full})}{\sigma_{proxy} \cdot \sigma_{full}}$$

其中$R$表示架构排名。

典型代理设置：
- ImageNet → ImageNet-100（100类子集）
- CIFAR-10 → 缩减训练集（10%）
- 训练epoch：完整训练的1/10

**2. 性能预测器**

基于部分训练曲线预测最终性能：

学习曲线建模：
$$\text{Acc}(t) = a - b \cdot t^{-c}$$

其中$a$是渐近精度，$b, c$是曲线参数。

基于前k个epoch预测：
$$\hat{a} = \argmin_a \sum_{i=1}^k ||\text{Acc}(i) - (a - b \cdot i^{-c})||^2$$

**3. 早期拒绝策略**

快速淘汰低质量架构：

贝叶斯优化框架：
$$\alpha(x) = \frac{\mu(x) - \xi}{\sigma(x)}$$

其中$\mu(x), \sigma(x)$是高斯过程的均值和方差，$\xi$是探索参数。

中位数剪枝：
- 训练到t epoch时，淘汰性能低于中位数的架构
- 资源重新分配给剩余架构
- 典型设置：t ∈ {10, 20, 30} epochs

### 25.1.7 可微分搜索的稳定性改进

**1. 离散化偏差问题**

DARTS等方法存在的问题：
- 连续松弛与离散化之间的gap
- 倾向选择参数量少的操作（如skip connection）

改进方法：

PC-DARTS（部分通道连接）：
$$\bar{o}^{(i,j)} = \sum_{o} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o'} \exp(\alpha_{o'}^{(i,j)})} \cdot o(x^{(i)}_{1/K})$$

只对1/K的通道进行架构搜索，减少内存消耗和过拟合。

**2. 公平性改进（FairDARTS）**

引入Sigmoid函数替代Softmax：
$$p_o = \sigma(\alpha_o) = \frac{1}{1 + \exp(-\alpha_o)}$$

独立选择每个操作，避免竞争导致的不公平。

**3. 鲁棒性增强（R-DARTS）**

引入扰动训练提高稳定性：

$$\min_\alpha \mathcal{L}_{val}(w^*(\alpha) + \epsilon, \alpha)$$

其中$\epsilon \sim \mathcal{N}(0, \sigma^2 I)$是高斯噪声。

Hessian正则化：
$$\mathcal{R}(\alpha) = ||\nabla^2_\alpha \mathcal{L}_{val}||_F$$

减少架构参数对验证损失的二阶敏感度。

### 25.1.8 大语言模型的NAS应用

**1. Transformer架构搜索空间**

针对LLM的搜索维度：
- 注意力头数：{4, 8, 12, 16}
- FFN扩展比：{2, 3, 4}
- 层数分配：不同阶段的深度
- 注意力模式：{全局, 局部, 稀疏}

**2. 自回归特性的考虑**

KV Cache优化的架构设计：

内存消耗建模：
$$M_{KV} = 2 \times L \times H \times D \times S \times B$$

其中：
- L：层数
- H：头数
- D：每头维度
- S：序列长度
- B：批大小

搜索时的约束：
$$M_{KV} + M_{weights} + M_{activation} \leq M_{total}$$

**3. 混合精度架构搜索**

不同层使用不同精度：

搜索空间扩展：
$$\mathcal{S} = \mathcal{S}_{arch} \times \mathcal{S}_{precision}$$

其中$\mathcal{S}_{precision} = \{INT4, INT8, FP16\}^L$

联合优化目标：
$$\min_{\alpha, \beta} -\text{PPL}(\alpha, \beta) + \lambda \cdot \text{BitOps}(\alpha, \beta)$$

其中$\beta$是精度配置，BitOps是位操作数。

**4. 实例：GPT模型的自动压缩**

搜索策略：
1. 固定总参数量预算（如7B）
2. 搜索层数、宽度、注意力配置
3. 考虑推理内存和计算效率
4. 使用困惑度（PPL）作为质量指标

典型发现：
- 浅而宽的架构在边缘设备上更高效
- 交替使用全局和局部注意力
- 早期层可以使用更低精度
- FFN可以比注意力层更激进地压缩

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

### 25.2.5 特定硬件的操作分解

**1. ARM NEON指令集优化**

ARM CPU的SIMD优化考虑：

向量化效率建模：
$$\eta_{vec} = \frac{\text{Theoretical SIMD Ops}}{\text{Actual Vector Ops}} \times \frac{\text{Vector Width}}{\text{Data Width}}$$

搜索空间约束：
- 通道数应为4的倍数（float32）或8的倍数（float16）
- 卷积核大小影响寄存器分配效率
- 深度可分离卷积的向量化模式：
  ```
  Depthwise: 每个向量处理4个空间位置
  Pointwise: 每个向量处理4个输出通道
  ```

内存访问模式优化：
$$\text{Cache Miss Rate} = 1 - \frac{\text{Reused Data}}{\text{Total Access}} \times \min(1, \frac{\text{Working Set}}{\text{Cache Size}})$$

**2. GPU Warp调度优化**

GPU特定的并行度考虑：

占用率（Occupancy）计算：
$$\text{Occupancy} = \frac{\text{Active Warps}}{\text{Max Warps}} = \frac{\text{Blocks} \times \text{Warps per Block}}{\text{SM Count} \times \text{Max Warps per SM}}$$

寄存器压力约束：
$$R_{per\_thread} \times T_{per\_block} \leq R_{max\_per\_SM}$$

共享内存约束：
$$S_{per\_block} \leq S_{max\_per\_SM}$$

搜索空间设计原则：
- Tile大小应匹配warp大小（32）
- 避免bank conflict：stride应与bank数互质
- 利用tensor core：维度应为8的倍数（INT8）或16的倍数（FP16）

**3. DSP向量处理器优化**

Hexagon HVX的特殊考虑：

向量寄存器宽度：1024位
$$\text{Elements per Vector} = \frac{1024}{\text{Bits per Element}}$$

VLIW并行度：
- 最多4条向量指令并行
- 2条标量指令
- 1条加载/存储指令

循环展开因子优化：
$$\text{Unroll Factor} = \min(\frac{\text{Vector Length}}{\text{Data Width}}, \frac{\text{Available Registers}}{2})$$

**4. NPU固定功能单元**

专用加速器的约束建模：

支持的操作集合：
$$\mathcal{O}_{NPU} = \{\text{Conv2D}, \text{DepthwiseConv2D}, \text{FC}, \text{Pool}, \text{Activation}\}$$

量化要求：
- 权重：INT8或INT4
- 激活：INT8或INT16
- 累加器：INT32

内存层次结构：
```
片上SRAM（快，小）→ 系统内存（中等）→ 外部存储（慢，大）
```

数据重用策略：
$$\text{Reuse} = \min(\frac{\text{SRAM Size}}{\text{Working Set}}, 1) \times \text{Temporal Locality}$$

### 25.2.6 内存带宽优化的架构设计

**1. Roofline模型在NAS中的应用**

计算强度定义：
$$I = \frac{\text{FLOPs}}{\text{Memory Bytes}}$$

性能上界：
$$P = \min(P_{peak}, I \times BW)$$

搜索空间中的操作分类：
- 计算密集型（I > $P_{peak}/BW$）：标准卷积
- 内存密集型（I < $P_{peak}/BW$）：深度卷积、激活函数

优化策略：
- 增加计算强度：使用更大的卷积核
- 减少内存访问：操作融合、重计算

**2. 数据布局感知的搜索**

不同布局的性能影响：
- NCHW：适合卷积操作
- NHWC：适合深度卷积和内存连续访问
- NC/HW/c：适合向量化处理

布局转换开销：
$$T_{transpose} = \frac{\text{Data Size}}{\text{Memory Bandwidth}} \times (1 + \text{Cache Miss Penalty})$$

搜索策略：
- 最小化布局转换次数
- 在关键路径上保持一致的数据布局
- 考虑硬件的原生布局偏好

**3. 操作融合机会识别**

可融合的操作模式：
- Conv + BatchNorm + ReLU
- Depthwise + Pointwise（MobileNet block）
- Multi-head attention计算

融合收益估算：
$$\text{Speedup} = \frac{T_{separate}}{T_{fused}} = \frac{\sum T_i + \sum T_{mem}}{T_{compute} + T_{mem\_fused}}$$

其中$T_{mem\_fused} < \sum T_{mem}$由于减少了中间结果的存储。

**4. 批处理与流水线设计**

动态批处理策略：
$$B_{opt} = \argmax_B \frac{B \times \text{Throughput}(B)}{\text{Latency}(B)}$$

满足内存约束：
$$B \times (\text{Activation Memory} + \text{KV Cache}) \leq \text{Available Memory}$$

流水线深度优化：
- 计算与数据传输重叠
- 多个请求的并行处理
- 考虑缓存局部性

### 25.2.7 编译器友好的架构设计

**1. 图优化机会**

搜索空间设计应考虑编译器优化：

常量折叠：
- 使用静态形状而非动态形状
- 固定的超参数（如组数、扩展因子）

算子融合模式：
- 垂直融合：连续的element-wise操作
- 水平融合：并行的相同操作
- 复合模式：预定义的高效kernel

死代码消除：
- 避免总是为0的分支
- 可静态确定的条件

**2. 量化友好的设计原则**

对称vs非对称量化：
$$Q(x) = \text{clip}(\text{round}(\frac{x}{s}), q_{min}, q_{max})$$

对称量化（zero-point = 0）：
- 硬件实现简单
- 适合权重量化

非对称量化：
- 更好的动态范围利用
- 适合激活量化

搜索空间考虑：
- 激活函数的值域（ReLU6 vs ReLU）
- BatchNorm的折叠可能性
- 残差连接的量化累积误差

**3. 内存分配优化**

静态内存规划：
$$M_{total} = \max_{t} \sum_{tensor \in Live(t)} \text{Size}(tensor)$$

其中$Live(t)$是时刻t的活跃张量集合。

内存复用策略：
- In-place操作优先
- 生命周期不重叠的张量共享内存
- 考虑对齐要求（通常16字节或32字节）

**4. 调度友好的拓扑结构**

并行执行机会：
- 多分支结构（如Inception）
- 独立的子图
- 异构执行（CPU+GPU）

依赖链长度：
$$\text{Critical Path} = \max_{\text{path}} \sum_{op \in path} T_{op}$$

搜索目标：
- 最小化关键路径长度
- 最大化并行度
- 平衡各执行单元负载

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

### 25.3.5 代理模型与贝叶斯优化

在NAS中直接评估每个架构的真实性能代价高昂，代理模型（Surrogate Model）提供了一种高效的性能预测方法。贝叶斯优化则利用这些预测来智能地探索搜索空间。

**1. 性能预测器设计**

基于架构特征的预测器：

编码架构为特征向量：
$$\mathbf{x} = [\text{depth}, \text{width}, \text{operators}, \text{connections}]$$

常用的预测模型：
- 高斯过程（GP）：提供不确定性估计
- 随机森林：处理离散特征
- 图神经网络：捕捉拓扑结构信息

训练数据收集策略：
- 随机采样：无偏但可能低效
- 主动学习：选择不确定性最大的架构
- 迁移学习：利用相关任务的数据

**2. 高斯过程回归**

GP建模架构性能：
$$f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$$

均值函数：
$$m(\mathbf{x}) = \mathbb{E}[f(\mathbf{x})]$$

协方差函数（核函数）：
$$k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{||\mathbf{x} - \mathbf{x}'||^2}{2l^2}\right)$$

后验预测：
$$\mu(\mathbf{x}_*) = \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}$$
$$\sigma^2(\mathbf{x}_*) = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*$$

**3. 贝叶斯优化框架**

获取函数（Acquisition Function）指导搜索：

期望改进（EI）：
$$\text{EI}(\mathbf{x}) = \mathbb{E}[\max(f(\mathbf{x}) - f^+, 0)]$$

其中$f^+$是当前最优值。

闭式解：
$$\text{EI}(\mathbf{x}) = \sigma(\mathbf{x})[\gamma \Phi(\gamma) + \phi(\gamma)]$$

其中：
- $\gamma = \frac{\mu(\mathbf{x}) - f^+}{\sigma(\mathbf{x})}$
- $\Phi$和$\phi$分别是标准正态分布的CDF和PDF

上置信界（UCB）：
$$\text{UCB}(\mathbf{x}) = \mu(\mathbf{x}) + \beta \sigma(\mathbf{x})$$

$\beta$控制探索与利用的平衡。

**4. 多保真度优化**

利用不同精度的评估降低成本：

保真度级别：
- 低保真度：少量epoch训练、小数据集
- 中保真度：中等训练、代理任务
- 高保真度：完整训练、真实任务

多保真度GP：
$$f(\mathbf{x}, z) \sim \mathcal{GP}(m(\mathbf{x}, z), k((\mathbf{x}, z), (\mathbf{x}', z')))$$

其中$z$表示保真度级别。

成本感知获取函数：
$$\text{EI/Cost}(\mathbf{x}, z) = \frac{\text{EI}(\mathbf{x}, z)}{C(z)^\alpha}$$

### 25.3.6 进化策略的现代改进

**1. 正则化进化（Regularized Evolution）**

基本思想：维护固定大小的种群，定期淘汰最老的个体。

算法流程：
```
1. 初始化种群P（大小S）
2. While not converged:
   a. 采样父代个体
   b. 应用变异操作
   c. 评估新个体
   d. 加入种群，移除最老个体
```

年龄正则化的优势：
- 避免早期优秀个体主导
- 保持种群多样性
- 自然的探索-利用平衡

**2. 锦标赛选择改进**

多目标锦标赛选择：
```
1. 随机选择k个个体
2. 按Pareto支配关系排序
3. 若存在非支配个体，随机选择一个
4. 否则，选择拥挤度最大的
```

自适应锦标赛大小：
$$k_t = k_{min} + (k_{max} - k_{min}) \cdot (1 - e^{-t/\tau})$$

早期使用小锦标赛促进探索，后期增大以加强选择压力。

**3. 变异算子设计**

架构感知的变异策略：
- 操作变异：随机改变某层的操作类型
- 连接变异：增加或删除跳跃连接
- 深度变异：插入或删除整层
- 宽度变异：改变通道数

变异率自适应：
$$p_m = p_{m,0} \cdot \exp(-\lambda \cdot \text{fitness\_variance})$$

当种群收敛（fitness方差小）时增加变异率。

**4. 协同进化策略**

将架构分解为多个子组件分别进化：

子种群设置：
- 种群1：编码器架构
- 种群2：解码器架构
- 种群3：连接模式

适应度共享：
$$f_{shared}(i) = \frac{f(i)}{\sum_{j} sh(d_{ij})}$$

其中共享函数：
$$sh(d) = \begin{cases}
1 - d/\sigma_{share} & \text{if } d < \sigma_{share} \\
0 & \text{otherwise}
\end{cases}$$

### 25.3.7 实时搜索与在线适应

**1. 增量式架构搜索**

动态调整已部署模型：

渐进式生长：
```
1. 从小模型开始
2. 监控资源使用和性能
3. 当有额外资源时，扩展架构
4. 在线微调新增部分
```

收缩策略：
- 识别低贡献度的组件
- 逐步减少其容量
- 监控性能下降
- 必要时恢复

**2. 上下文感知搜索**

根据运行时条件调整：

多模式架构：
$$\mathcal{A} = \{A_{low}, A_{med}, A_{high}\}$$

模式选择策略：
$$A_t = \begin{cases}
A_{high} & \text{if } P_{battery} > 50\% \land L_{cpu} < 30\% \\
A_{med} & \text{if } 20\% < P_{battery} \leq 50\% \\
A_{low} & \text{otherwise}
\end{cases}$$

平滑切换机制：
- 共享骨干网络
- 仅切换可变部分
- 使用知识蒸馏保持一致性

**3. 联邦架构搜索**

分布式设备上的协同搜索：

本地搜索：
- 每个设备在本地数据上搜索
- 考虑设备特定的约束
- 定期上传架构和性能

全局聚合：
$$A_{global} = \argmax_A \sum_{i} w_i \cdot \text{Score}_i(A)$$

其中$w_i$反映设备i的重要性（数据量、可靠性等）。

隐私保护机制：
- 仅分享架构参数，不分享数据
- 使用差分隐私添加噪声
- 安全聚合协议

### 25.3.8 自动化超参数优化

NAS过程本身有许多超参数需要调整，自动化这一过程可以提高搜索效率。

**1. 搜索超参数的优化**

关键超参数：
- 学习率调度：$\eta_t = \eta_0 \cdot \cos(\frac{\pi t}{T})$
- 权重衰减：$\lambda \in [10^{-5}, 10^{-2}]$
- 架构参数温度：$\tau \in [0.1, 1.0]$
- 正则化系数：硬件感知项的权重

嵌套优化：
$$\begin{aligned}
\min_{\Lambda} & \quad \mathcal{L}_{val}(\alpha^*(\Lambda), w^*(\alpha^*, \Lambda)) \\
\text{s.t.} & \quad \alpha^* = \argmin_\alpha \mathcal{L}_{search}(\alpha, w^*; \Lambda) \\
& \quad w^* = \argmin_w \mathcal{L}_{train}(w, \alpha; \Lambda)
\end{aligned}$$

**2. 自适应搜索策略**

根据搜索进展动态调整：

收敛检测：
$$\text{Converged} = \frac{||\mathcal{P}_t - \mathcal{P}_{t-\Delta t}||}{||\mathcal{P}_t||} < \epsilon$$

其中$\mathcal{P}_t$是时刻t的Pareto前沿。

策略切换：
- 早期：重视多样性，大变异率
- 中期：平衡探索与利用
- 后期：局部精细搜索

**3. 元学习加速**

利用历史搜索经验：

任务相似度：
$$\text{Sim}(T_i, T_j) = \exp(-||\phi(T_i) - \phi(T_j)||^2)$$

其中$\phi$提取任务特征（数据集统计、硬件规格等）。

初始化策略：
$$\theta_0 = \sum_{i} \text{Sim}(T_{new}, T_i) \cdot \theta_i^*$$

使用相似任务的最优解加权初始化新任务。

**4. 搜索空间自动设计**

基于性能分析自动构建搜索空间：

操作重要性评分：
$$I(op) = \mathbb{E}_{\alpha \in \mathcal{A}}[\text{Perf}(\alpha) | op \in \alpha] - \mathbb{E}_{\alpha \in \mathcal{A}}[\text{Perf}(\alpha)]$$

自动剪枝低价值操作：
- 移除重要性分数低的操作
- 保留互补的操作组合
- 根据硬件特性调整

层次化搜索空间：
1. 宏观搜索：整体架构模式
2. 微观搜索：块内部结构
3. 精细搜索：具体超参数

通过分阶段搜索降低复杂度。

## 25.4 自动化压缩流程

将NAS与其他压缩技术（量化、剪枝、知识蒸馏）结合，可以构建端到端的自动化模型压缩流程。这种集成方法能够充分发挥各种技术的优势，获得更高的压缩率和更好的性能。

### 25.4.1 联合优化框架

**1. 统一的优化目标**

综合考虑架构、量化、剪枝的联合优化：

$$\min_{\alpha, q, m} \mathcal{L}_{task}(\alpha, q, m) + \lambda_1 \cdot \text{Size}(\alpha, q, m) + \lambda_2 \cdot \text{Latency}(\alpha, q, m)$$

其中：
- $\alpha$：架构参数
- $q$：量化配置（位宽分配）
- $m$：剪枝掩码

模型大小计算：
$$\text{Size} = \sum_{l} \frac{b_l \cdot c_{in,l} \cdot c_{out,l} \cdot k_l^2 \cdot (1-s_l)}{8 \times 1024^2} \text{ MB}$$

其中$b_l$是层$l$的位宽，$s_l$是稀疏度。

**2. 交替优化策略**

由于联合优化空间巨大，通常采用交替优化：

```
1. 固定q, m，优化架构α
2. 固定α, m，优化量化配置q
3. 固定α, q，优化剪枝掩码m
4. 重复直到收敛
```

每个子问题的求解：
- 架构优化：使用NAS方法（DARTS、演化等）
- 量化优化：基于敏感度分析的位宽分配
- 剪枝优化：重要性评分 + 结构化约束

**3. 端到端可微分框架**

将所有压缩技术统一到可微分框架中：

可微分量化：
$$\tilde{w} = s \cdot \text{round}(\frac{w}{s})$$

使用直通估计器（STE）进行梯度回传：
$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial \tilde{w}}$$

可微分剪枝：
$$\tilde{w} = w \odot \sigma(\alpha_m \cdot g)$$

其中$g$是重要性分数，$\alpha_m$控制剪枝程度。

### 25.4.2 硬件感知的压缩策略

**1. 平台特定的压缩配置**

不同硬件对压缩技术的支持差异：

ARM CPU优化：
- INT8量化：NEON指令集原生支持
- 结构化剪枝：保持缓存友好的访问模式
- 通道剪枝粒度：4的倍数（向量化考虑）

GPU优化：
- 2:4稀疏：Ampere架构的硬件加速
- FP16/INT8混合精度：Tensor Core利用
- 块稀疏模式：匹配warp执行模型

NPU/DSP优化：
- 固定点量化：通常仅支持INT8/INT16
- 规则稀疏模式：硬件加速单元要求
- 层融合：减少片外内存访问

**2. 延迟感知的压缩决策**

理论压缩率不等于实际加速比：

有效压缩率：
$$\text{Effective Ratio} = \frac{\text{Original Latency}}{\text{Compressed Latency}}$$

延迟预测模型：
$$T_{compressed} = T_{compute} \cdot (1 - s) \cdot \frac{b_{compressed}}{b_{original}} + T_{overhead}$$

其中$T_{overhead}$包含解压缩、格式转换等开销。

**3. 内存层次感知**

考虑不同存储层次的带宽和容量：

分层存储策略：
- L1缓存：存储最频繁访问的权重
- L2缓存：存储当前层的完整权重
- 主内存：存储整个模型

压缩决策影响：
```
if (compressed_size < L2_cache_size):
    # 整层可以缓存，激进压缩
    quantization_bits = 4
else:
    # 需要频繁内存访问，保守压缩
    quantization_bits = 8
```

### 25.4.3 渐进式压缩流水线

**1. 多阶段压缩策略**

逐步增加压缩强度，保持模型质量：

阶段1：架构搜索
- 寻找高效的基础架构
- 不考虑量化和剪枝
- 优化FLOPs和参数量

阶段2：混合精度量化
- 基于敏感度分析分配位宽
- 保持关键层的高精度
- 量化感知训练

阶段3：结构化剪枝
- 识别冗余通道/层
- 保持硬件友好的结构
- 微调恢复性能

阶段4：联合优化
- 同时调整所有压缩参数
- 精细化调整
- 最终部署准备

**2. 知识蒸馏辅助**

使用教师模型指导压缩过程：

多教师蒸馏：
$$\mathcal{L}_{KD} = \sum_{i} \alpha_i \cdot KL(p_{student} || p_{teacher_i})$$

其中不同教师代表不同压缩阶段的模型。

特征对齐：
$$\mathcal{L}_{feature} = \sum_{l} \beta_l \cdot ||f_l^{student} - \phi_l(f_l^{teacher})||^2$$

$\phi_l$是特征变换函数，处理维度不匹配。

**3. 自适应压缩强度**

根据任务难度动态调整：

任务复杂度估计：
- 类别数量
- 数据集大小
- 类间相似度
- 噪声水平

压缩强度映射：
$$\text{Compression Ratio} = f(\text{Task Complexity}, \text{Hardware Constraints})$$

简单任务可以更激进地压缩，复杂任务需要保守。

### 25.4.4 实例：视觉Transformer的自动压缩

**1. ViT特定的压缩挑战**

Vision Transformer的独特结构带来新的压缩机会：

Token剪枝：
- 识别不重要的图像patch
- 动态减少序列长度
- 保持关键区域的分辨率

注意力头剪枝：
$$\text{Importance}_h = \frac{1}{N} \sum_{n=1}^{N} ||\text{Attention}_h^{(n)}||_F$$

剪除重要性低的注意力头。

层跳跃（Layer Skipping）：
- 浅层用于简单样本
- 深层用于复杂样本
- 动态路由机制

**2. 分辨率自适应**

根据内容复杂度调整输入分辨率：

复杂度评分：
$$C_{img} = \text{Entropy}(img) + \lambda \cdot \text{EdgeDensity}(img)$$

分辨率选择：
$$r = \begin{cases}
224 \times 224 & \text{if } C_{img} > \tau_{high} \\
160 \times 160 & \text{if } \tau_{low} < C_{img} \leq \tau_{high} \\
112 \times 112 & \text{otherwise}
\end{cases}$$

**3. 混合专家（MoE）压缩**

将大模型压缩为多个专家的混合：

专家分配：
$$p(e|x) = \text{softmax}(W_g \cdot x)$$

稀疏激活：
- 每个样本只激活k个专家
- 降低计算成本
- 保持模型容量

负载均衡损失：
$$\mathcal{L}_{balance} = \sum_{e} \text{Var}(\text{Load}_e)$$

确保专家负载均匀，避免某些专家过度使用。

### 25.4.5 压缩质量评估与验证

**1. 多维度评估指标**

综合评估压缩效果：

压缩率指标：
- 模型大小压缩比：$\frac{\text{Original Size}}{\text{Compressed Size}}$
- FLOPs减少率：$1 - \frac{\text{Compressed FLOPs}}{\text{Original FLOPs}}$
- 内存占用减少：包括权重和激活

性能保持率：
$$\text{Performance Retention} = \frac{\text{Compressed Accuracy}}{\text{Original Accuracy}} \times 100\%$$

硬件效率提升：
- 实测推理延迟减少
- 能耗降低百分比
- 吞吐量提升倍数

**2. 鲁棒性验证**

压缩模型的鲁棒性测试：

分布偏移测试：
- 不同光照条件
- 各种噪声级别
- 对抗样本攻击

量化误差累积分析：
$$\epsilon_{total} = \sum_{l=1}^{L} \epsilon_l \cdot \prod_{j=l+1}^{L} ||W_j||$$

确保误差不会在深层网络中爆炸。

**3. 部署前验证**

实际部署环境测试：

边缘设备测试矩阵：
```
设备类型 | 批大小 | 延迟要求 | 通过率
---------|--------|----------|--------
手机     | 1      | <50ms    | 98%
平板     | 4      | <100ms   | 95%
嵌入式   | 1      | <200ms   | 99%
```

长时间运行稳定性：
- 内存泄漏检查
- 温度监控
- 功耗一致性

### 25.4.6 工具链集成与自动化

**1. 压缩工作流编排**

使用配置文件定义压缩流程：

```yaml
compression_pipeline:
  - stage: nas
    config:
      search_space: mobilenet_v3
      target_latency: 50ms
      hardware: snapdragon_865
  
  - stage: quantization
    config:
      method: mixed_precision
      calibration_samples: 1000
      target_model_size: 10MB
  
  - stage: pruning
    config:
      sparsity: 0.5
      structure: channel
      granularity: 4
  
  - stage: optimization
    config:
      compiler: tensorrt
      precision: int8
      workspace_size: 1GB
```

**2. 持续集成/部署（CI/CD）**

自动化测试和部署：

触发条件：
- 模型架构更新
- 新硬件平台支持
- 性能回归检测

自动化流程：
1. 拉取最新模型
2. 执行压缩pipeline
3. 运行性能基准测试
4. 生成压缩报告
5. 部署到边缘设备
6. 收集真实世界反馈

**3. 压缩策略版本管理**

跟踪和管理不同的压缩配置：

版本控制内容：
- 搜索空间定义
- 压缩超参数
- 硬件配置文件
- 评估指标历史

A/B测试框架：
- 并行部署多个压缩版本
- 收集性能对比数据
- 自动选择最优配置

通过这种系统化的方法，可以将模型压缩从手工调优转变为自动化、可重复的工程流程。

## 本章小结

本章深入探讨了神经架构搜索（NAS）在边缘推理场景中的应用。我们学习了如何设计边缘友好的搜索空间，理解了硬件感知搜索的重要性，掌握了多目标优化的各种策略，并了解了如何将NAS与其他压缩技术结合形成自动化的模型优化流程。

**关键要点：**

1. **边缘导向的搜索空间设计**
   - 优先选择硬件友好的操作（深度可分离卷积、组卷积等）
   - 考虑量化和部署的约束
   - 平衡模型容量与硬件限制

2. **硬件感知的优化策略**
   - 使用真实硬件延迟而非FLOPs作为优化目标
   - 考虑内存层次结构和带宽限制
   - 针对特定硬件特性（SIMD、GPU warp、NPU限制）定制搜索

3. **多目标优化技术**
   - Pareto前沿分析帮助理解精度-效率权衡
   - 演化算法与梯度方法的结合
   - 约束优化处理硬性限制

4. **自动化压缩流程**
   - NAS、量化、剪枝、蒸馏的联合优化
   - 渐进式压缩保持模型质量
   - 端到端的自动化工具链

**核心公式回顾：**

搜索空间的硬件成本建模：
$$\text{Cost} = \alpha \cdot \text{FLOPs} + \beta \cdot \text{Memory Access} + \gamma \cdot \text{Latency}$$

多目标优化的Pareto支配关系：
$$x \prec y \iff \forall i: f_i(x) \leq f_i(y) \land \exists j: f_j(x) < f_j(y)$$

贝叶斯优化的期望改进：
$$\text{EI}(\mathbf{x}) = \sigma(\mathbf{x})[\gamma \Phi(\gamma) + \phi(\gamma)]$$

联合压缩的优化目标：
$$\min_{\alpha, q, m} \mathcal{L}_{task} + \lambda_1 \cdot \text{Size} + \lambda_2 \cdot \text{Latency}$$

## 练习题

### 基础题

1. **搜索空间设计**
   设计一个适用于ARM Cortex-A78 CPU的NAS搜索空间，要求：
   - 基本操作包括3×3和5×5深度可分离卷积
   - 考虑NEON向量化（128位宽）
   - 通道数必须是4的倍数
   
   *提示：考虑哪些操作在ARM NEON上有高效实现*

2. **延迟预测模型**
   给定一个卷积操作：输入[1, 224, 224, 32]，卷积核3×3，输出通道64，步长1。
   计算在以下条件下的理论延迟：
   - 峰值计算能力：10 GFLOPS
   - 内存带宽：25.6 GB/s
   - 假设计算和内存访问不能重叠
   
   *提示：分别计算compute-bound和memory-bound时间，取最大值*

3. **Pareto前沿分析**
   给定5个架构的性能数据：
   - A: 精度72%, 延迟15ms
   - B: 精度70%, 延迟10ms
   - C: 精度75%, 延迟25ms
   - D: 精度71%, 延迟12ms
   - E: 精度73%, 延迟20ms
   
   找出Pareto前沿上的架构。
   
   *提示：检查每个架构是否被其他架构支配*

4. **量化感知的架构搜索**
   设计一个搜索策略，同时优化架构和量化配置：
   - 架构选择：层数∈{12, 16, 20}，宽度倍数∈{0.5, 0.75, 1.0}
   - 量化选择：每层可选INT4或INT8
   - 约束：模型大小<10MB，延迟<50ms
   
   *提示：考虑如何编码搜索空间和评估函数*

### 挑战题

5. **硬件特定优化**
   针对Qualcomm Hexagon DSP设计NAS策略：
   - HVX向量单元：1024位宽
   - 支持的数据类型：INT8, INT16
   - VLIW并行：4条向量指令
   
   设计搜索空间和优化策略，说明如何利用这些硬件特性。
   
   *提示：考虑向量化效率、数据对齐、指令级并行*

6. **多保真度贝叶斯优化**
   设计一个三级保真度的NAS系统：
   - 低保真度：10个epoch，10%数据
   - 中保真度：50个epoch，50%数据  
   - 高保真度：200个epoch，100%数据
   
   如何设计获取函数来平衡探索成本和预测准确性？
   
   *提示：考虑成本加权的期望改进，以及何时切换保真度级别*

7. **实时自适应架构**
   设计一个可以根据运行时条件动态调整的模型架构：
   - 检测电池电量、CPU负载、内存压力
   - 支持3种运行模式：高性能、平衡、省电
   - 模式切换延迟<100ms
   
   描述架构设计和切换机制。
   
   *提示：考虑共享backbone、可选分支、快速切换策略*

8. **端到端压缩pipeline**
   为一个100M参数的视觉Transformer设计完整的压缩流程，目标是部署到移动GPU上：
   - 原始模型：ViT-Base, 86M参数，精度78%
   - 目标：<10M参数，延迟<30ms，精度>72%
   - 可用技术：NAS、混合精度量化、结构化剪枝、知识蒸馏
   
   设计压缩策略的执行顺序和每步的具体方法。
   
   *提示：考虑各技术的互补性、压缩顺序对最终效果的影响*

<details>
<summary>答案</summary>

1. ARM搜索空间应包含：深度可分离卷积（3×3, 5×5），通道数{16, 32, 64, 128}，使用ReLU6激活，包含残差连接。

2. 计算时间 = 51.4M FLOPs / 10 GFLOPS = 5.14ms；内存时间 = 7.4MB / 25.6GB/s = 0.29ms；理论延迟 = 5.14ms

3. Pareto前沿：B（70%, 10ms）、A（72%, 15ms）、E（73%, 20ms）、C（75%, 25ms）

4. 使用超网络方法，架构参数和量化参数联合训练，通过硬件感知的损失函数引导搜索

5. 搜索空间应包含1024位对齐的张量操作，优先使用可向量化的操作，设计VLIW友好的并行模式

6. 使用多保真度GP建模，获取函数考虑信息增益/成本比，当低保真度不确定性降低后切换到高保真度

7. 采用超网络设计，共享前几层，后续层根据模式选择不同宽度，使用渐进式切换避免突变

8. 顺序：1)NAS找到高效架构 2)知识蒸馏到目标架构 3)混合精度量化 4)通道剪枝微调 5)编译器优化

</details>
