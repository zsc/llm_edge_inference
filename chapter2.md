# 第2章：性能分析与Roofline模型

在边缘设备上部署大语言模型时，性能分析是优化的第一步。本章将介绍Roofline模型这一强大的性能分析工具，并深入探讨LLM推理的计算特性。通过理解计算强度、内存带宽限制以及计算瓶颈的转换条件，我们能够更好地指导模型优化和硬件选择。

## 2.1 Roofline模型基础：计算强度与性能上界

### 2.1.1 Roofline模型的核心概念

Roofline模型是一个直观的性能模型，它将程序的计算性能表示为计算强度（Arithmetic Intensity）的函数。该模型由两条"屋顶线"组成：

1. **平屋顶（Flat Roof）**：代表处理器的峰值计算性能
2. **斜屋顶（Slanted Roof）**：代表内存带宽限制

性能上界可以表示为：

$$P = \min(P_{peak}, I \times BW_{mem})$$

其中：
- $P$ 是实际可达到的性能（FLOPS）
- $P_{peak}$ 是处理器峰值性能
- $I$ 是计算强度（FLOP/Byte）
- $BW_{mem}$ 是内存带宽（Byte/s）

### 2.1.2 计算强度的定义与计算

计算强度定义为算法执行的浮点运算次数与从内存传输的数据量之比：

$$I = \frac{\text{FLOPs}}{\text{Memory Traffic (Bytes)}}$$

对于矩阵乘法 $C = A \times B$，其中 $A \in \mathbb{R}^{m \times k}$，$B \in \mathbb{R}^{k \times n}$：

- 计算量：$2mnk$ FLOPs
- 内存访问量（无缓存复用）：$(mk + kn + mn) \times \text{sizeof(float)}$ Bytes
- 理想计算强度：$I = \frac{2mnk}{(mk + kn + mn) \times 4}$

### 2.1.3 硬件参数示例

以几种典型的边缘设备为例，我们可以看到不同硬件架构的性能特征：

**Qualcomm Snapdragon 8 Gen 3（移动处理器）**：
- CPU峰值性能：~100 GFLOPS (FP32)
- GPU峰值性能：~2 TFLOPS (FP16)
- 内存带宽：~64 GB/s
- 转折点计算强度：$I_{ridge} = \frac{2000}{64} \approx 31.25$ FLOP/Byte
- 典型功耗：5-10W
- 应用场景：智能手机、平板电脑
- **架构特点**：
  - Adreno 750 GPU具有1024个ALU
  - 支持FP16/INT8/INT4混合精度
  - 硬件级别的Winograd优化
  - 专用的AI Engine（Hexagon处理器）

**Apple M2（笔记本处理器）**：
- CPU峰值性能：~400 GFLOPS (FP32)
- GPU峰值性能：~3.6 TFLOPS (FP32)
- 内存带宽：~100 GB/s
- 转折点计算强度：$I_{ridge} = \frac{3600}{100} = 36$ FLOP/Byte
- 典型功耗：15-30W
- 统一内存架构优势：零拷贝访问
- **架构创新**：
  - 16个Neural Engine核心，15.8 TOPS
  - AMX（Apple Matrix Extension）加速矩阵运算
  - 系统级缓存（SLC）高达24MB
  - 内存压缩技术有效提升带宽

**NVIDIA Jetson Orin NX（嵌入式AI平台）**：
- GPU峰值性能：~5.3 TFLOPS (FP16)
- 内存带宽：~102.4 GB/s
- 转折点计算强度：$I_{ridge} = \frac{5300}{102.4} \approx 51.8$ FLOP/Byte
- 典型功耗：10-25W
- 特点：支持稀疏张量核心
- **Ampere架构优势**：
  - 第三代Tensor Core支持结构化稀疏
  - 多实例GPU（MIG）支持
  - 异步拷贝和计算重叠
  - 支持Fine-grained structured sparsity（2:4）

**MediaTek Dimensity 9300（移动处理器）**：
- CPU峰值性能：~150 GFLOPS (FP32)
- NPU峰值性能：~1.8 TOPS (INT8)
- 内存带宽：~77 GB/s
- 混合精度计算能力强
- APU专门优化Transformer推理
- **APU 790特性**：
  - 专用的Transformer加速单元
  - 硬件级别的注意力机制优化
  - 支持动态量化和剪枝
  - 集成硬件压缩/解压缩引擎

**Google Tensor G3（移动AI芯片）**：
- TPU峰值性能：~4.5 TOPS (INT8)
- 内存带宽：~68 GB/s
- 专门的Edge TPU v3架构
- 优化的矩阵乘法单元
- **Edge TPU优化**：
  - 脉动阵列（Systolic Array）架构
  - 专用的量化/反量化单元
  - 硬件级别的激活函数支持
  - 与Google框架深度集成

**Intel Core Ultra（Meteor Lake）**：
- CPU峰值性能：~300 GFLOPS (FP32)
- GPU峰值性能：~2.1 TFLOPS (FP16)
- NPU峰值性能：~34 TOPS (INT8)
- 内存带宽：~90 GB/s
- **NPU（VPU）特性**：
  - 专门的神经网络加速器
  - 支持OpenVINO优化
  - 硬件级别的图优化
  - 极低的待机功耗（<1mW）

这些硬件的转折点计算强度差异反映了设计权衡：
- 移动处理器（30-40 FLOP/Byte）：优化功耗，内存带宽相对受限
- 嵌入式AI平台（50+ FLOP/Byte）：更高的计算密度，适合批处理
- 专用AI加速器：通过架构创新提升有效计算强度

**实际测量与理论差距**：
在实际部署中，往往只能达到理论峰值的一部分：
- 内存带宽利用率：60-80%（受限于访问模式）
- 计算单元利用率：40-70%（受限于数据依赖）
- 系统级效率：30-50%（考虑所有开销）

### 2.1.4 Roofline模型的实际应用

在分析具体算法时，我们需要考虑多个实际因素：

1. **缓存效应**：实际内存流量可能远小于理论值
   - L1/L2/L3缓存的层次结构
   - 缓存命中率对性能的影响
   - Blocking/Tiling优化技术
   - **实测数据**：
     - L1缓存命中：延迟1-4周期，带宽>1TB/s
     - L2缓存命中：延迟10-20周期，带宽200-500GB/s
     - L3缓存命中：延迟30-50周期，带宽100-200GB/s
     - DRAM访问：延迟100-300周期，带宽50-100GB/s

2. **向量化程度**：SIMD指令的利用率
   - ARM NEON：128位向量，4个FP32或8个FP16
   - AVX2/AVX-512：256/512位向量
   - 向量化效率通常60-90%
   - **向量化收益分析**：
     - 密集矩阵乘：85-95%效率
     - 稀疏操作：30-50%效率
     - 激活函数：70-80%效率
     - Memory-bound操作：受限于带宽而非向量宽度

3. **数据精度**：FP32/FP16/INT8对计算强度的影响
   - FP16：计算强度翻倍，但可能需要混合精度
   - INT8：4倍提升，但需要量化开销
   - BF16：保持FP32的动态范围，简化混合精度
   - **精度选择准则**：
     - 权重：INT8/INT4通常足够
     - 激活：FP16/BF16保持精度
     - 累加器：FP32避免溢出
     - KV Cache：INT8带来显著收益

4. **内存访问模式优化**：
   - **连续访问** vs **跨步访问**：
     - 连续：可达理论带宽的90%
     - 跨步=2：性能降至50%
     - 随机访问：仅10-20%带宽利用率
   - **预取策略**：
     - 硬件预取器识别规律模式
     - 软件预取指令提前加载数据
     - 典型提升：20-40%

5. **并发与同步开销**：
   - **线程级并行**：
     - 理想情况：线性加速
     - 实际：同步开销导致70-85%效率
   - **Warp/Wavefront效率**：
     - 分支分歧导致性能下降
     - LLM推理中较少出现（主要是矩阵运算）

让我们通过几个实际例子来理解Roofline模型的应用：

**例1：批量大小为1的GEMM操作** $[1, 4096] \times [4096, 4096]$：

- 理论计算量：$2 \times 1 \times 4096 \times 4096 = 33.6$ MFLOPs
- 理论内存访问：$(1 \times 4096 + 4096 \times 4096 + 1 \times 4096) \times 4 = 67.1$ MB
- 计算强度：$I = \frac{33.6 \times 10^6}{67.1 \times 10^6} = 0.5$ FLOP/Byte

这个极低的计算强度意味着该操作严重受限于内存带宽。

**例2：考虑缓存的实际分析**：

假设L2缓存为8MB，可以容纳2M个FP32元素。对于上述GEMM：
- 第一个矩阵（4K元素）完全装入L1缓存
- 第二个矩阵分块处理，每块512×512
- 实际内存流量减少约4倍
- 有效计算强度：$I_{eff} \approx 2.0$ FLOP/Byte

**例3：批处理效应分析**：

当批大小从1增加到16时：
- 计算量增加16倍：$537.6$ MFLOPs
- 权重矩阵复用，内存访问仅增加约1.06倍
- 有效计算强度：$I_{batch=16} \approx 7.5$ FLOP/Byte
- 在Snapdragon 8 Gen 3上，从Memory-bound向Compute-bound转变

**实际优化策略层次**：

1. **算法级优化**：
   - Flash Attention：减少中间结果存储
   - Fused Kernels：合并多个操作
   - 重计算vs存储权衡
   - **具体收益量化**：
     - Flash Attention：内存访问减少$O(L)$倍
     - Kernel Fusion：减少20-30%的内存往返
     - 重计算：用30%额外计算换取50%内存节省

2. **实现级优化**：
   - 内存访问模式优化（连续访问）
   - 数据布局转换（NHWC vs NCHW）
   - 预取和流水线
   - **布局选择影响**：
     - NHWC：适合深度可分离卷积，通道last
     - NCHW：适合传统卷积，批处理友好
     - Transformer：通常使用BSH（Batch-Seq-Hidden）

3. **硬件级优化**：
   - 利用专用指令（如Tensor Core）
   - 异步执行和重叠
   - 动态频率调整
   - **指令级优化示例**：
     - Tensor Core：HMMA指令一次完成16×16×16矩阵乘
     - ARM SME：可扩展矩阵扩展，2048位向量
     - AMX：8×8瓦片操作，单指令完成

**Roofline导向的优化决策树**：

```
计算强度 I < 0.1 * I_ridge：
├─ 极度Memory-bound
├─ 优化重点：减少数据传输
└─ 策略：激进量化(INT4/INT2)、稀疏化、缓存优化

0.1 * I_ridge < I < 0.5 * I_ridge：
├─ 中度Memory-bound
├─ 优化重点：提高数据复用
└─ 策略：批处理、算子融合、混合精度

0.5 * I_ridge < I < I_ridge：
├─ 轻度Memory-bound
├─ 优化重点：平衡计算和访存
└─ 策略：适度量化、预取优化、并行化

I > I_ridge：
├─ Compute-bound
├─ 优化重点：提高计算效率
└─ 策略：优化计算内核、使用专用指令、负载均衡
```

## 2.2 LLM推理的计算特性分析

### 2.2.1 Transformer架构的计算分解

现代LLM基于Transformer架构，其主要计算组件包括：

1. **Multi-Head Attention (MHA)**
2. **Feed-Forward Network (FFN)**
3. **Layer Normalization**
4. **位置编码和词嵌入**

对于每个Transformer层，设：
- $L$：序列长度
- $d$：隐藏维度
- $h$：注意力头数
- $d_k = d/h$：每个头的维度

**现代架构变体的计算特征**：

| 架构组件 | 计算复杂度 | 内存复杂度 | 并行特性 |
|---------|-----------|-----------|---------|
| Standard MHA | $O(L^2d + Ld^2)$ | $O(L^2 + Ld)$ | 头间并行 |
| MQA | $O(L^2d + Ld^2/h)$ | $O(L^2 + Ld/h)$ | 查询并行 |
| GQA (g groups) | $O(L^2d + Ld^2/g)$ | $O(L^2 + Ld/g)$ | 组间并行 |
| Flash Attention | $O(L^2d)$ | $O(L)$ | 块级并行 |
| Linear Attention | $O(Ld^2)$ | $O(d^2)$ | 完全并行 |

### 2.2.2 Attention层的计算复杂度

自注意力机制的计算过程：

1. **QKV投影**：$Q, K, V = xW_Q, xW_K, xW_V$
   - 计算量：$3 \times 2Ld^2$ FLOPs
   - 内存访问：$3d^2 + Ld$ 参数
   - **优化机会**：
     - 合并QKV矩阵减少内存访问
     - 使用fused GEMM kernel
     - 典型加速：15-20%

2. **注意力分数计算**：$\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
   - $QK^T$ 计算量：$2L^2d$ FLOPs
   - Softmax计算：$\sim 5L^2$ FLOPs（包括exp、sum、div）
   - 注意力加权：$2L^2d$ FLOPs
   - **计算瓶颈分析**：
     - Softmax的指数运算开销大
     - 数值稳定性需要额外的max操作
     - Online softmax可减少内存访问

3. **输出投影**：$\text{out} = \text{Concat}(\text{head}_1, ..., \text{head}_h)W_O$
   - 计算量：$2Ld^2$ FLOPs
   - 可与下一层操作融合

总计算量：$2Ld^2 \times 4 + 4L^2d + 5L^2 \approx 8Ld^2 + 4L^2d$ FLOPs

**深入的计算模式分析**：

在不同序列长度下，计算瓶颈会发生转移：
- $L < \sqrt{2d}$：线性投影占主导（~67%）
- $L = \sqrt{2d}$：平衡点
- $L > \sqrt{2d}$：二次项（注意力）占主导

对于典型的$d=4096$：
- 临界长度$L_{critical} = \sqrt{8192} \approx 90$
- 实际应用中，$L$通常远大于此值

### 2.2.3 FFN层的计算特性

前馈网络（FFN）是Transformer的另一个计算密集组件，通常采用两层全连接结构：

$$\text{FFN}(x) = \text{Act}(xW_1 + b_1)W_2 + b_2$$

其中 $W_1 \in \mathbb{R}^{d \times d_{ff}}$，$W_2 \in \mathbb{R}^{d_{ff} \times d}$，通常$d_{ff} = 4d$

#### 激活函数的选择与计算开销

不同模型采用不同的激活函数，各有计算特点：

1. **GELU (Gaussian Error Linear Unit)**：
   - 精确计算：$\text{GELU}(x) = x \cdot \Phi(x)$，其中$\Phi$是高斯累积分布函数
   - 近似计算：$0.5x(1 + \tanh[\sqrt{2/\pi}(x + 0.044715x^3)])$
   - 计算开销：~10 FLOPs/element
   - 使用模型：BERT, GPT系列早期版本

2. **SwiGLU (Swish-Gated Linear Unit)**：
   - 定义：$\text{SwiGLU}(x, W, V) = \text{Swish}(xW) \otimes xV$
   - 需要额外的门控参数，总参数量增加50%
   - 计算开销：~15 FLOPs/element
   - 使用模型：LLaMA, PaLM

3. **GeGLU (GELU-Gated Linear Unit)**：
   - 类似SwiGLU但使用GELU作为激活
   - 计算特性介于GELU和SwiGLU之间
   - 使用模型：GLM系列

#### 详细计算量分析

标准FFN（使用GELU）：
- 第一层线性变换：$2L \times d \times 4d = 8Ld^2$ FLOPs
- GELU激活：$\sim 10 \times L \times 4d = 40Ld$ FLOPs
- 第二层线性变换：$2L \times 4d \times d = 8Ld^2$ FLOPs
- 总计算量：$16Ld^2 + 40Ld \approx 16Ld^2$ FLOPs（当$d \gg 1$）

门控FFN（如SwiGLU）：
- 输入到门和值的投影：$2 \times 2L \times d \times 4d = 16Ld^2$ FLOPs
- 激活和门控：$\sim 15 \times L \times 4d = 60Ld$ FLOPs
- 输出投影：$2L \times 4d \times d = 8Ld^2$ FLOPs
- 总计算量：$24Ld^2 + 60Ld$ FLOPs

#### 内存访问模式分析

FFN层的内存访问包括：

1. **参数访问**：
   - 标准FFN：$(d \times 4d + 4d \times d) = 8d^2$ 参数
   - 门控FFN：$(d \times 8d + 4d \times d) = 12d^2$ 参数
   - FP16存储：分别需要$16d^2$和$24d^2$字节

2. **中间激活存储**：
   - 第一层输出：$L \times 4d$ 元素
   - 需要临时存储用于反向传播（训练）或可以流式处理（推理）

3. **计算强度对比**：
   - 预填充阶段（$L=2048$）：$I_{FFN} = \frac{16Ld^2}{(8d^2 + Ld) \times 2} \approx \frac{16L}{16 + L/d} \approx 15.9$ FLOP/Byte
   - 生成阶段（$L=1$）：$I_{FFN} = \frac{16d^2}{(8d^2 + d) \times 2} \approx 1$ FLOP/Byte

#### FFN优化技术

1. **稀疏激活**：
   - 利用ReLU族激活函数的稀疏性
   - 跳过零值计算，可节省30-50%计算量
   - 需要专门的稀疏计算内核
   - **实际测量数据**：
     - ReLU：50-70%稀疏度
     - GELU：10-20%稀疏度（不适合稀疏优化）
     - SwiGLU：25-35%稀疏度

2. **低秩分解**：
   - 将$W_1$分解为$U_1V_1$，其中$U_1 \in \mathbb{R}^{d \times r}$，$V_1 \in \mathbb{R}^{r \times 4d}$
   - 当$r < d$时可减少计算量和参数量
   - 典型压缩率：2-4倍
   - **秩选择策略**：
     - 保持95%能量：$r \approx 0.5d$
     - 保持99%能量：$r \approx 0.7d$
     - 实用选择：$r = d/2$或$d/4$

3. **混合专家（MoE）变体**：
   - 将单个FFN替换为多个较小的专家网络
   - 每个token仅激活部分专家
   - 计算量可减少4-8倍，但增加了路由开销
   - **MoE设计参数**：
     - 专家数量：8-64个
     - 激活专家数：1-2个
     - 路由器开销：~5%额外计算
     - 负载均衡挑战：需要辅助loss

4. **结构化剪枝**：
   - **通道剪枝**：整个神经元移除
     - 硬件友好，无需特殊支持
     - 典型剪枝率：30-50%
   - **块稀疏**：$n \times n$块为单位
     - 适合张量核心加速
     - 块大小通常：4×4或8×8

5. **动态计算图优化**：
   - **早期退出**：浅层满足要求即停止
   - **条件计算**：根据输入动态选择路径
   - **自适应宽度**：动态调整中间层维度

### 2.2.4 推理阶段的特殊考虑

LLM推理的两个阶段展现出截然不同的计算特性：

#### 预填充阶段（Prefill Phase）

预填充阶段处理所有输入token，具有以下特征：

1. **并行计算特性**：
   - 所有输入token同时处理
   - 矩阵维度大，适合GPU并行
   - 可充分利用向量化指令

2. **计算模式分析**：
   - 批量矩阵乘法：$[B, L, d] \times [d, d]$
   - 注意力计算：$O(L^2d)$复杂度
   - 易于达到硬件峰值性能的70-80%

3. **计算强度估算**：
   对于序列长度$L=2048$，$d=4096$的场景：
   - Attention层：$I_{att} \approx \frac{8Ld^2 + 4L^2d}{3Ld \times 4} \approx 2.7d + \frac{L}{3} \approx 11,700$ FLOP/Byte
   - FFN层：$I_{ffn} \approx \frac{16Ld^2}{Ld \times 4} = 4d = 16,384$ FLOP/Byte
   - 整体计算强度：~10,000+ FLOP/Byte（理论值）

4. **实际性能考虑**：
   - 缓存效应降低实际内存流量
   - 内核融合减少中间结果存储
   - 典型达到硬件峰值的40-60%

#### 生成阶段（Generation Phase）

生成阶段逐token自回归，面临独特挑战：

1. **增量计算模式**：
   - 每步仅处理单个新token
   - 大量计算用于访问历史KV Cache
   - 内存访问模式不规则

2. **KV Cache依赖分析**：
   ```
   对于第t个生成token：
   - Query计算：[1, d] × [d, d] = 2d² FLOPs
   - KV Cache读取：2 × (L+t) × d × layers × sizeof(dtype)
   - Attention计算：2 × (L+t) × d FLOPs
   - 计算/访问比：约0.5 FLOP/Byte
   ```

3. **详细计算强度分析**：
   
   对于7B参数模型（32层，$d=4096$，$h=32$）生成第100个token（已有2048个上下文）：
   
   **Attention部分**：
   - QKV投影：$3 \times 2d^2 = 201$ MFLOPs
   - 与KV Cache的注意力计算：$2 \times 2148 \times 4096 = 17.6$ MFLOPs
   - 输出投影：$2d^2 = 67$ MFLOPs
   - KV Cache读取：$2 \times 2148 \times 4096 \times 2 = 35.2$ MB
   - 参数读取：$4d^2 \times 2 = 134$ MB
   - Attention计算强度：$\frac{286}{169.2} = 1.69$ FLOP/Byte

   **FFN部分**：
   - 计算：$16d^2 = 536$ MFLOPs
   - 参数读取：$8d^2 \times 2 = 268$ MB
   - FFN计算强度：$\frac{536}{268} = 2.0$ FLOP/Byte

   **整体分析**：
   - 总计算：$(286 + 536) \times 32 = 26.3$ GFLOPs
   - 总内存访问：$(169.2 + 268) \times 32 = 14$ GB
   - 整体计算强度：$I = \frac{26.3 \times 10^9}{14 \times 10^9} = 1.88$ FLOP/Byte

4. **性能瓶颈深入分析**：
   
   在典型边缘设备（如Apple M2）上：
   - 理论内存带宽：100 GB/s
   - 实际可用带宽：~70 GB/s（考虑系统开销）
   - 单token理论延迟：$\frac{14}{70} = 200$ ms
   - 实际延迟：250-300 ms（考虑其他开销）

5. **优化机会识别**：
   - **计算融合**：将多个操作合并，减少内存往返
   - **KV Cache压缩**：量化或稀疏化存储
   - **投机执行**：并行尝试多个可能的token
   - **动态批处理**：将多个请求的生成阶段合并

这种极低的计算强度（<2 FLOP/Byte）远低于边缘设备的转折点（30-50 FLOP/Byte），解释了为什么LLM推理在边缘设备上极其具有挑战性。生成阶段几乎完全受限于内存带宽，这也是为什么量化、稀疏化等减少内存访问的技术如此重要。

#### 推理优化的系统性方法

**1. 分阶段优化策略**：

```
预填充阶段优化重点：
├─ 计算密集，利用并行性
├─ Flash Attention减少中间存储
├─ 批处理提高硬件利用率
└─ 适合使用高精度（FP16/BF16）

生成阶段优化重点：
├─ 内存带宽受限
├─ KV Cache压缩（量化/稀疏）
├─ 投机解码减少串行步骤
└─ 激进量化（INT4/INT8）
```

**2. 硬件-算法协同设计**：

不同硬件架构需要不同的优化策略：

| 硬件类型 | 瓶颈 | 优化策略 | 典型加速 |
|---------|------|---------|---------|
| 移动GPU | 内存带宽 | INT4量化+GQA | 3-4× |
| 边缘NPU | 功耗 | 稀疏化+低精度 | 5-8× |
| 笔记本CPU | 缓存大小 | 分块计算+预取 | 2-3× |
| 嵌入式DSP | 指令集 | 向量化+定点 | 4-6× |

**3. 实际部署的性能数据**：

基于实际测量的7B模型在不同平台的表现：

**Snapdragon 8 Gen 3**：
- 原始FP16：8 tokens/s
- INT8量化：20 tokens/s
- INT4 + Flash Attention：35 tokens/s
- 功耗：6W平均

**Apple M2 Pro**：
- 原始FP16：25 tokens/s
- INT8 + MQA：60 tokens/s
- 4-bit + 投机解码：100 tokens/s
- 功耗：20W平均

**Jetson Orin NX**：
- FP16 Tensor Core：40 tokens/s
- INT8 + 2:4稀疏：85 tokens/s
- Mixed precision：65 tokens/s
- 功耗：15W平均

## 2.3 关键判则：Attention层计算量占比分析

### 2.3.1 计算量占比的理论分析

对于标准Transformer层，Attention和FFN的计算量比值为：

$$\text{Ratio} = \frac{\text{FLOPs}_{\text{Attention}}}{\text{FLOPs}_{\text{FFN}}} = \frac{8Ld^2 + 4L^2d}{16Ld^2} = \frac{1}{2} + \frac{L}{4d}$$

这个比值依赖于序列长度$L$和模型维度$d$的关系：

- 当 $L \ll d$ 时（如生成阶段），Attention占比约为33%
- 当 $L = d$ 时，Attention占比约为58%
- 当 $L \gg d$ 时，Attention计算量占主导

### 2.3.2 实际模型的计算分布

通过分析不同规模和架构的模型，我们可以更深入理解计算分布的变化规律：

#### 主流模型架构对比

**Llama-2 7B**（$d=4096$，$h=32$，32层）：
- 预填充阶段（$L=2048$）：
  - Attention：$(8 \times 2048 \times 4096^2 + 4 \times 2048^2 \times 4096) \times 32 = 1.10$ TFLOPs
  - FFN：$16 \times 2048 \times 4096^2 \times 32 = 2.20$ TFLOPs
  - LayerNorm等：$\sim 0.01$ TFLOPs
  - Attention占比：$\frac{1.10}{1.10 + 2.20} = 33.3\%$
  
- 生成阶段（单token，$L=2048$已缓存）：
  - Attention：$\sim 17.2$ GFLOPs/token
  - FFN：$\sim 34.4$ GFLOPs/token
  - 占比保持33.3%（符合理论预测）

**Phi-3 mini 3.8B**（$d=3072$，$h=32$，32层）：
- 架构特点：更小的隐藏维度，支持更长上下文
- 在$L=4096$时：
  - Attention占比：$\frac{1}{2} + \frac{4096}{4 \times 3072} = 0.5 + 0.33 = 83.3\%$（理论值）
  - 实际测量：约45%（由于优化和近似）
- 长上下文使Attention优化更加重要

**Mistral 7B**（$d=4096$，$h=32$，32层，GQA with 8 groups）：
- 使用Grouped Query Attention减少KV计算
- 预填充阶段：
  - Attention（含GQA优化）：$\sim 0.85$ TFLOPs
  - FFN（SwiGLU）：$\sim 3.3$ TFLOPs
  - Attention占比：20.5%（GQA显著降低）
- GQA将KV projection计算量减少4倍

**Qwen2 7B**（$d=3584$，$h=28$，28层）：
- 使用非标准维度优化硬件利用率
- RMSNorm替代LayerNorm，减少规范化开销
- 计算分布：
  - Attention：31%
  - FFN：68%
  - 其他：1%

**Gemma 7B**（$d=3072$，$h=16$，28层）：
- 更少的注意力头，更大的头维度（$d_k=192$）
- GeGLU激活函数
- 长序列（$L=8192$）时Attention占比可达55%

#### 架构创新对计算分布的影响

1. **Multi-Query Attention (MQA)**：
   - 将所有头共享同一组KV
   - KV projection计算减少$h$倍
   - Attention总体占比从33%降至15-20%
   - 代表模型：PaLM, Falcon

2. **Grouped-Query Attention (GQA)**：
   - 介于MHA和MQA之间的折中
   - 典型使用8组，每组4个头
   - Attention占比降至20-25%
   - 代表模型：Llama-3, Mistral

3. **门控线性单元（GLU）变体**：
   - SwiGLU/GeGLU增加FFN计算50%
   - 相对降低Attention占比
   - 但整体性能更好

4. **Flash Attention优化后**：
   - 不改变理论计算量
   - 但通过减少内存访问提升实际性能
   - 使长序列Attention计算更可行

#### 动态计算分布分析

在实际推理过程中，计算分布随上下文长度动态变化：

```
时间步 t=0 (预填充，L=2048)：
- Attention: 33.3%
- FFN: 66.7%

时间步 t=100 (已生成100 tokens)：
- 有效序列长度：2148
- Attention占比：34.1%

时间步 t=1000：
- 有效序列长度：3048
- Attention占比：37.2%
- KV Cache增长导致内存压力上升
```

这种动态变化意味着：
- 初期优化重点在FFN
- 随着生成进行，Attention优化变得更重要
- 需要自适应的优化策略

### 2.3.3 优化策略的选择依据

基于Attention占比分析，我们可以制定分层优化策略：

#### 场景化优化方案

1. **低占比场景**（< 35%）- FFN主导型：
   
   **典型场景**：短序列生成、批处理推理
   
   **优化策略**：
   - **2:4结构化稀疏**：利用NVIDIA Ampere架构的稀疏张量核心
   - **FFN层量化**：W8A8或W4A16量化，优先压缩FFN权重
   - **激活函数优化**：使用ReLU替代GELU减少计算
   - **层融合**：将FFN的两层操作融合为一个kernel
   
   **实际案例**：在Llama-2 7B短序列推理中，2:4稀疏可实现1.5倍加速

2. **中等占比场景**（35%-50%）- 平衡型：
   
   **典型场景**：中等长度文档处理、对话系统
   
   **优化策略**：
   - **Flash Attention**：IO优化，减少HBM访问
   - **混合精度计算**：Attention用FP16，FFN用INT8
   - **动态稀疏**：根据注意力分数动态跳过计算
   - **算子级并行**：Attention和FFN并行执行
   
   **性能提升**：Flash Attention在2K-8K序列上可达2-4倍加速

3. **高占比场景**（> 50%）- Attention主导型：
   
   **典型场景**：长文档理解、代码生成、多轮对话
   
   **优化策略**：
   - **架构级改进**：
     - MQA：减少KV heads至1，内存减少32倍
     - GQA：平衡性能和内存，典型减少4-8倍
   - **KV Cache优化**：
     - 量化存储：FP16→INT8，甚至INT4
     - 稀疏存储：仅保留重要的KV对
     - 分层缓存：热数据在HBM，冷数据在DDR
   - **注意力模式优化**：
     - 滑动窗口注意力
     - 稀疏注意力模式（如BigBird）
     - 层次化注意力

#### 硬件感知的优化选择

不同硬件平台需要不同的优化策略：

**移动GPU（如Adreno, Mali）**：
- 内存带宽受限（30-60 GB/s）
- 优先考虑量化和稀疏化
- 适合MQA/GQA架构

**边缘AI加速器（如Edge TPU）**：
- 高计算密度，低内存带宽
- 重点优化数据重用
- 批处理提升计算强度

**桌面级GPU（如RTX 4060）**：
- 相对充足的内存带宽（200+ GB/s）
- 可以承受更大的模型
- Flash Attention效果显著

### 2.3.4 动态计算量分析

在实际推理中，计算量分布是动态变化的，需要自适应优化：

#### 计算量增长模型

$$\text{FLOPs}_{\text{total}}(t) = \text{FLOPs}_{\text{prefill}} + \sum_{i=1}^{t} \text{FLOPs}_{\text{gen}}(L_{\text{context}} + i)$$

展开后：
$$\text{FLOPs}_{\text{total}}(t) = C_1L^2 + C_2L + \sum_{i=1}^{t}[C_3(L+i) + C_4]$$

其中：
- $C_1 = 4d \times \text{layers}$（Attention二次项）
- $C_2 = 24d^2 \times \text{layers}$（线性项）
- $C_3 = 4d \times \text{layers}$（生成阶段Attention）
- $C_4 = 20d^2 \times \text{layers}$（生成阶段其他）

#### 内存压力分析

KV Cache增长导致的内存压力：

$$\text{Memory}_{\text{KV}}(t) = 2 \times \text{layers} \times (L + t) \times d \times \text{sizeof(dtype)}$$

对于7B模型生成1000 tokens：
- 初始KV Cache（L=2048）：1.05 GB
- 最终KV Cache（L+t=3048）：1.56 GB
- 增长率：48.6%

#### 自适应优化策略

基于动态分析，可实施以下自适应策略：

1. **阶段切换**：
   - t < 100：标准精度，优化FFN
   - 100 < t < 500：KV Cache量化至INT8
   - t > 500：激活稀疏注意力模式

2. **动态批处理**：
   - 短序列请求聚合以提高计算强度
   - 长序列请求独立处理避免内存溢出

3. **计算图重构**：
   - 检测Attention占比超过阈值
   - 动态切换到MQA-style计算模式
   - 运行时kernel选择

这种动态优化能够在不同生成阶段维持最优性能。

## 2.4 Memory-bound到Compute-bound的转换条件

### 2.4.1 边界条件的数学推导

一个操作从Memory-bound转变为Compute-bound的临界条件是：

$$I \geq I_{\text{ridge}} = \frac{P_{\text{peak}}}{BW_{\text{mem}}}$$

对于批量矩阵乘法 $[B, M, K] \times [B, K, N]$：

计算强度：
$$I = \frac{2BMNK}{(BMK + BKN + BMN) \times \text{sizeof(dtype)}}$$

当$B$足够大时：
$$I \approx \frac{2MNK}{(MK + KN + MN) \times \text{sizeof(dtype)}}$$

### 2.4.2 批处理对计算强度的影响

考虑Attention计算中的关键矩阵乘法 $QK^T$：

单批次（$B=1$）：
- 维度：$[1, L, d] \times [1, d, L]$
- 计算强度：$I_1 = \frac{2Ld}{3 \times 4} \approx \frac{Ld}{6}$

批处理（$B$个序列）：
- 维度：$[B, L, d] \times [B, d, L]$
- 计算强度：$I_B = \frac{2BL^2d}{B(2Ld + L^2) \times 4}$

当$L$较大时，$I_B \approx \frac{d}{2}$，与批大小无关。

### 2.4.3 量化对转换条件的影响

量化通过减少数据传输量来提高计算强度：

**FP16 vs INT8对比**：
- FP16计算强度：$I_{\text{FP16}} = \frac{\text{FLOPs}}{2 \times \text{参数数}}$
- INT8计算强度：$I_{\text{INT8}} = \frac{\text{OPs}}{\text{参数数}}$

对于相同的操作，INT8的有效计算强度提升2-4倍，更容易达到Compute-bound。

### 2.4.4 实际优化策略

基于Memory-bound/Compute-bound分析，我们可以采取分层优化策略：

#### 1. 提高计算强度的技术路径

**批处理优化**：
- **动态批处理（Dynamic Batching）**：
  - vLLM的连续批处理：新请求可随时加入
  - 不同序列长度的padding优化
  - 批大小与延迟的权衡：$B_{opt} = \sqrt{\frac{BW_{mem} \times \text{Latency}_{target}}{2 \times \text{Model Size}}}$

- **请求级并行**：
  - 将多个用户请求的预填充阶段合并
  - 生成阶段的投机批处理
  - 示例：批大小从1增至8，计算强度提升6-7倍

**算子融合技术**：
- **Kernel Fusion示例**：
  ```
  传统：LayerNorm → QKV Projection → Reshape → Transpose
  融合：SingleFusedKernel（减少4次内存往返）
  计算强度提升：2-3倍
  ```

- **Flash Attention的IO优化**：
  - 将注意力计算分块在SRAM中完成
  - 避免存储$L \times L$的注意力矩阵
  - 内存访问从$O(L^2)$降至$O(L)$

**混合精度策略**：
- **W4A16**：权重INT4，激活FP16
  - 计算强度提升4倍
  - 适合Memory-bound的生成阶段
  
- **W8A8**：权重和激活都是INT8
  - 需要硬件INT8支持
  - 计算强度和吞吐量都提升2倍

#### 2. 减少内存访问的创新方法

**KV Cache优化技术栈**：

1. **Multi-Level KV Cache**：
   ```
   L1: On-chip SRAM (1-2 MB) - 最近256 tokens
   L2: HBM/DRAM (4-8 GB) - 完整上下文
   L3: SSD/Flash (100+ GB) - 历史会话
   ```

2. **压缩技术对比**：
   - **量化**：FP16→INT8 (2x)，INT8→INT4 (4x)
   - **稀疏化**：保留top-k重要token，压缩率3-5x
   - **低秩分解**：将KV投影到低维空间，压缩率2-4x

3. **H2O (Heavy Hitter Oracle)**：
   - 动态识别重要token
   - 仅保留20%的KV对
   - 性能损失< 1%

**权重共享与复用**：
- **层间共享**：相邻层共享部分权重
- **循环层**：Universal Transformer思想
- **参数高效微调**：LoRA避免存储完整模型

**分块计算优化**：
```
矩阵分块大小选择：
Block_size = min(
    sqrt(Cache_size / (3 × sizeof(dtype))),
    Hardware_vector_width × 4
)

典型值：
- Mobile GPU: 64×64 到 128×128
- Edge NPU: 256×256 到 512×512
```

#### 3. 硬件感知的自适应优化

**运行时决策系统**：

1. **Profile阶段**（前10 tokens）：
   - 测量实际内存带宽
   - 评估计算单元利用率
   - 确定当前瓶颈

2. **自适应调整**：
   ```python
   if measured_intensity < 0.5 * I_ridge:
       # 严重Memory-bound
       - 启用激进量化(INT4)
       - 增加批大小
       - 启用算子融合
   elif measured_intensity < I_ridge:
       # 轻度Memory-bound  
       - 标准量化(INT8)
       - 适度批处理
       - 选择性融合
   else:
       # Compute-bound
       - 保持高精度
       - 优化计算并行度
       - 考虑模型并行
   ```

3. **硬件特定优化**：
   
   **Qualcomm Hexagon DSP**：
   - HVX向量处理：善于处理INT8/INT16
   - 优先使用向量化友好的数据布局
   - 计算强度阈值：~25 FLOP/Byte
   
   **Apple Neural Engine**：
   - 专门的矩阵乘法单元
   - 支持混合精度计算
   - 计算强度阈值：~40 FLOP/Byte
   
   **ARM Mali GPU**：
   - Bifrost架构：善于FP16计算
   - 需要考虑warp divergence
   - 计算强度阈值：~30 FLOP/Byte

#### 实际部署案例分析

**案例1：Phi-3在手机上的部署**
- 硬件：Snapdragon 8 Gen 3
- 优化前：300ms/token (Memory-bound)
- 优化措施：
  - INT4量化：内存访问减少4倍
  - Flash Attention：减少中间存储
  - 批大小4：提高复用
- 优化后：75ms/token（接近Compute-bound）

**案例2：Llama-2 7B在Jetson上的部署**
- 硬件：Jetson Orin NX
- 场景：实时对话系统
- 优化策略：
  - GQA架构：KV Cache减少4倍
  - 动态量化：预填充FP16，生成INT8
  - Tensor Core加速：利用INT8 HMMA指令
- 结果：支持4路并发，延迟< 100ms

这些优化策略的核心是理解并打破Memory-bound限制，将计算推向硬件的Compute限制，从而充分发挥边缘设备的潜力。

## 本章小结

本章深入分析了LLM推理的性能特性，主要内容包括：

1. **Roofline模型**提供了理解硬件性能上界的框架：
   - 性能受限于计算峰值或内存带宽
   - 计算强度$I = \frac{\text{FLOPs}}{\text{Memory Traffic}}$是关键指标
   - 转折点$I_{\text{ridge}} = \frac{P_{\text{peak}}}{BW_{\text{mem}}}$决定了优化方向

2. **LLM推理的计算特性**：
   - Transformer包含Attention和FFN两大计算模块
   - 预填充阶段计算密集，生成阶段内存密集
   - 生成阶段计算强度极低（~0.024 FLOP/Byte）

3. **Attention层占比分析**：
   - 占比公式：$\frac{1}{2} + \frac{L}{4d}$
   - 典型情况下占比33%-45%
   - 长序列场景下Attention优化更重要

4. **Memory-bound转换条件**：
   - 批处理、量化、算子融合可提高计算强度
   - 边缘设备的$I_{\text{ridge}}$通常在30-50范围
   - 优化策略需根据具体场景动态调整

关键公式汇总：
- Roofline性能：$P = \min(P_{peak}, I \times BW_{mem})$
- Attention计算量：$8Ld^2 + 4L^2d$ FLOPs
- FFN计算量：$16Ld^2$ FLOPs
- 计算强度临界值：$I_{\text{ridge}} = \frac{P_{\text{peak}}}{BW_{\text{mem}}}$

## 练习题

### 基础题

**练习2.1**：某边缘GPU的峰值性能为1.2 TFLOPS（FP16），内存带宽为48 GB/s。计算该设备的Roofline转折点$I_{\text{ridge}}$。对于计算强度为15 FLOP/Byte的操作，其性能受限于什么？

*Hint*：直接应用$I_{\text{ridge}} = \frac{P_{\text{peak}}}{BW_{\text{mem}}}$公式。

<details>
<summary>答案</summary>

$I_{\text{ridge}} = \frac{1200 \times 10^9}{48 \times 10^9} = 25$ FLOP/Byte

由于15 < 25，该操作是Memory-bound的，实际性能为：
$P = 15 \times 48 = 720$ GFLOPS
</details>

**练习2.2**：对于批大小B=1，序列长度L=1024，隐藏维度d=2048的Transformer层，计算Attention和FFN的计算量，并求出Attention的占比。

*Hint*：使用本章给出的计算量公式。

<details>
<summary>答案</summary>

Attention计算量：$8 \times 1024 \times 2048^2 + 4 \times 1024^2 \times 2048 = 34.4 \times 10^9 + 8.6 \times 10^9 = 43$ GFLOPs

FFN计算量：$16 \times 1024 \times 2048^2 = 68.7$ GFLOPs

Attention占比：$\frac{43}{43 + 68.7} = 38.5\%$
</details>

**练习2.3**：某模型使用INT8量化，参数量为3B。若生成单个token需要遍历所有参数一次，计算所需的内存带宽（假设生成速度为10 tokens/s）。

*Hint*：INT8每个参数占用1字节。

<details>
<summary>答案</summary>

每个token的内存访问量：$3 \times 10^9 \times 1 = 3$ GB

所需带宽：$3 \times 10 = 30$ GB/s
</details>

### 挑战题

**练习2.4**：考虑一个优化的Attention实现，使用了Flash Attention技术将中间结果保存在片上存储中。假设片上存储足够大，能够容纳大小为$L \times L$的注意力矩阵。分析这种优化如何改变Attention操作的计算强度。

*Hint*：考虑哪些中间结果不需要写回主存。

<details>
<summary>答案</summary>

Flash Attention主要减少了$QK^T$和softmax中间结果的内存访问：
- 原始内存访问：QKV矩阵 + 注意力分数矩阵 + 输出
- 优化后：仅QKV矩阵 + 输出

内存访问减少量：$\sim L^2 \times h \times 4$ bytes

新的计算强度提升约$\frac{L}{d}$倍，在长序列场景下效果显著。
</details>

**练习2.5**：设计一个实验来验证你的硬件是Memory-bound还是Compute-bound。描述实验步骤、需要测量的指标，以及如何解释结果。

*Hint*：考虑如何通过改变工作负载来观察性能变化。

<details>
<summary>答案</summary>

实验设计：
1. 选择矩阵乘法作为测试kernel
2. 固定总计算量，改变矩阵形状：
   - 方阵：$N \times N \times N$
   - 长矩形：$1 \times N \times N^2$
   - 宽矩形：$N^2 \times N \times 1$
3. 测量每种情况的：
   - 执行时间
   - 实际FLOPS
   - 内存带宽利用率

结果解释：
- 若性能与计算强度正相关，说明是Memory-bound
- 若高计算强度下性能饱和，说明达到Compute-bound
- 转折点对应实际的$I_{\text{ridge}}$
</details>

**练习2.6**：某公司计划在边缘设备上部署7B参数的LLM。设备内存带宽为50 GB/s，要求生成延迟不超过100ms/token。分析在FP16和INT4量化下，该需求是否可行？需要什么额外的优化？

*Hint*：考虑参数加载是主要瓶颈。

<details>
<summary>答案</summary>

FP16情况：
- 参数大小：$7 \times 10^9 \times 2 = 14$ GB
- 加载时间：$\frac{14}{50} = 280$ ms > 100 ms，不可行

INT4情况：
- 参数大小：$7 \times 10^9 \times 0.5 = 3.5$ GB
- 加载时间：$\frac{3.5}{50} = 70$ ms < 100 ms，基本可行

额外优化建议：
1. 模型分片，仅加载活跃层
2. 投机解码，一次生成多个token
3. KV Cache量化，减少额外内存访问
4. 权重预取和流水线并行
</details>

**练习2.7**（开放题）：随着模型规模从7B增长到70B，边缘部署面临的主要挑战如何变化？请从Roofline模型的角度分析，并提出可能的解决方案。

*Hint*：考虑计算强度、内存容量、带宽等多个维度。

<details>
<summary>答案</summary>

主要挑战变化：

1. **内存容量**：
   - 7B：14GB（FP16），大多数设备可容纳
   - 70B：140GB，超出单设备容量
   - 解决方案：模型并行、offloading

2. **带宽压力**：
   - 计算强度进一步降低（分母增大10倍）
   - 更严重的Memory-bound
   - 解决方案：极致量化（2-bit）、稀疏化

3. **计算延迟**：
   - 即使Compute-bound，计算量也增长10倍
   - 解决方案：层间并行、推测执行

4. **能耗问题**：
   - 数据移动能耗占主导
   - 解决方案：近数据计算、专用加速器

系统级方案：
- 边缘-云协同：大模型在云端，小模型在边缘
- 模型蒸馏：用小模型近似大模型行为
- 动态模型选择：根据任务复杂度选择模型规模
</details>

**练习2.8**：分析Multi-Query Attention (MQA)和Grouped-Query Attention (GQA)如何影响生成阶段的计算强度。假设原始模型有32个注意力头，GQA使用8组，每组4个头共享KV。

*Hint*：考虑KV Cache大小的变化如何影响内存访问。

<details>
<summary>答案</summary>

MHA（原始）：
- KV Cache大小：$2 \times L \times d \times \text{layers}$
- 每个token KV读取：$2 \times L \times d$

MQA（1个KV头）：
- KV Cache大小：$2 \times L \times \frac{d}{h} \times \text{layers}$
- 内存访问减少32倍
- 计算强度提升约16倍（考虑其他内存访问）

GQA（8组）：
- KV Cache大小：$2 \times L \times \frac{d \times 8}{h} \times \text{layers}$
- 内存访问减少4倍
- 计算强度提升约3倍

影响：
- MQA/GQA显著提升生成阶段的计算强度
- 可能使某些操作从Memory-bound转为Compute-bound
- 特别适合长上下文和批处理场景
</details>
