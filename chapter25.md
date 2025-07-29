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
