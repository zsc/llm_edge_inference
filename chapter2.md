# 第2章：性能分析与Roofline模型

## 开篇

在边缘设备上部署大语言模型时，理解硬件性能瓶颈至关重要。本章介绍Roofline性能模型，并通过数学分析建立LLM推理的关键性能判则，帮助读者在系统设计时做出正确的优化决策。

## 2.1 Roofline模型基础：计算强度与性能上界

### 2.1.1 基本概念

Roofline模型通过两个关键指标描述程序性能上界：

**计算强度（Arithmetic Intensity, AI）**：
$$AI = \frac{\text{FLOPs}}{\text{Memory Traffic (Bytes)}}$$

**性能上界**：
$$P_{max} = \min(P_{peak}, AI \times BW_{mem})$$

其中：
- $P_{peak}$：峰值计算性能（FLOPs/s）
- $BW_{mem}$：内存带宽（Bytes/s）
- $AI$：算法的计算强度（FLOPs/Byte）

### 2.1.2 Roofline图解析

在对数坐标系中，Roofline模型呈现为：

1. **Memory-bound区域**（$AI < \frac{P_{peak}}{BW_{mem}}$）：
   - 性能受限于内存带宽
   - $P = AI \times BW_{mem}$

2. **Compute-bound区域**（$AI \geq \frac{P_{peak}}{BW_{mem}}$）：
   - 性能受限于计算能力
   - $P = P_{peak}$

**转折点**：
$$AI_{critical} = \frac{P_{peak}}{BW_{mem}}$$

### 2.1.3 典型硬件参数

| 设备类型 | 峰值性能 (TFLOPs) | 内存带宽 (GB/s) | $AI_{critical}$ |
|---------|------------------|----------------|----------------|
| Apple M1 | 2.6 (FP32) | 68.25 | 38.1 |
| Apple M2 | 3.6 (FP32) | 100 | 36.0 |
| NVIDIA Jetson Orin | 5.3 (FP32) | 204.8 | 25.9 |
| Snapdragon 8 Gen 3 | 0.75 (FP32) | 64 | 11.7 |

## 2.2 LLM推理的计算特性分析

### 2.2.1 Prefill阶段计算强度

对于序列长度$s$，模型维度$d$，层数$L$：

**Self-Attention计算量**：
- QKV投影：$3 \times s \times d^2$ FLOPs
- 注意力分数：$2 \times s^2 \times d$ FLOPs
- 输出投影：$s \times d^2$ FLOPs

**FFN计算量**：
- 上投影：$s \times d \times 4d = 4sd^2$ FLOPs
- 下投影：$s \times 4d \times d = 4sd^2$ FLOPs

**总计算量**（单层）：
$$FLOPs_{prefill} = 4sd^2 + 2s^2d + 8sd^2 = 12sd^2 + 2s^2d$$

**内存访问量**（假设权重已缓存）：
$$Mem_{prefill} = s \times d \times sizeof(float) \times \text{读写次数}$$

**Prefill计算强度**：
$$AI_{prefill} = \frac{12sd^2 + 2s^2d}{C \times s \times d \times 4} \approx \frac{3d + 0.5s}{C}$$

其中$C$为常数，取决于具体实现（通常$C \approx 4-8$）。

### 2.2.2 Decode阶段计算强度

每个token的计算：

**计算量**（单层）：
$$FLOPs_{decode} = 12d^2 + 2sd$$

**内存访问**（需要加载所有权重）：
$$Mem_{decode} = (12d^2 + \text{KV cache}) \times sizeof(float)$$

**Decode计算强度**：
$$AI_{decode} = \frac{12d^2 + 2sd}{12d^2 \times 4 + 2sd \times 4} \approx \frac{1}{4}$$

### 2.2.3 关键观察

1. Prefill阶段：计算强度随序列长度$s$增加
2. Decode阶段：计算强度极低，几乎总是memory-bound
3. 批处理可以显著提高decode的计算强度

## 2.3 关键判则：Attention层计算量分析

### 2.3.1 Attention计算量占比

**判则1：Attention计算量显著性**

当满足以下条件时，Attention计算不可忽略：
$$s > \frac{6d}{h}$$

其中$h$是注意力头数。

**推导**：
- Attention FLOPs：$2s^2d$
- FFN FLOPs：$8sd^2$
- 占比：$\frac{2s^2d}{2s^2d + 8sd^2} = \frac{s}{s + 4d}$

当占比超过20%时：$\frac{s}{s + 4d} > 0.2 \Rightarrow s > d$

### 2.3.2 实际模型参数

| 模型 | 隐藏维度$d$ | 头数$h$ | 临界序列长度 |
|------|------------|--------|-------------|
| LLaMA-7B | 4096 | 32 | 768 |
| LLaMA-13B | 5120 | 40 | 768 |
| GPT-3 175B | 12288 | 96 | 768 |

**经验法则**：
- 短序列（$s < 512$）：FFN主导
- 中等序列（$512 < s < 2048$）：混合瓶颈
- 长序列（$s > 2048$）：Attention主导

### 2.3.3 Flash Attention的影响

Flash Attention通过分块计算减少内存访问：
$$Mem_{flash} = O(\frac{s^2d}{M}) + O(sd)$$

其中$M$是SRAM大小。这将Attention的有效计算强度提升至：
$$AI_{flash} \approx \frac{2s^2d}{sd/M + sd} = \frac{2s}{1 + 1/M}$$

## 2.4 Memory-bound到Compute-bound的转换条件

### 2.4.1 批大小对计算强度的影响

对于批大小$B$的decode阶段：

**计算量**：
$$FLOPs_{batch} = B \times (12d^2 + 2sd)$$

**内存访问**（权重共享）：
$$Mem_{batch} = 12d^2 \times 4 + B \times \text{激活内存}$$

**批处理计算强度**：
$$AI_{batch} = \frac{B \times (12d^2 + 2sd)}{12d^2 \times 4 + B \times C \times d \times 4}$$

### 2.4.2 临界批大小计算

**判则2：Decode阶段compute-bound转换**

临界批大小：
$$B_{critical} = \frac{12d^2 \times (AI_{critical} - 1/4)}{C \times d \times AI_{critical}}$$

对于典型硬件（$AI_{critical} \approx 30$）：
$$B_{critical} \approx \frac{12d \times 29.75}{30C} \approx \frac{12d}{C}$$

### 2.4.3 实例分析

以Apple M2为例（$AI_{critical} = 36$）：

| 模型 | $d$ | 预估$B_{critical}$ | 实测值 |
|------|-----|------------------|-------|
| LLaMA-7B | 4096 | ~6000 | 5500 |
| LLaMA-13B | 5120 | ~7500 | 7000 |

**实用建议**：
1. $B < 100$：强memory-bound，优化内存访问
2. $100 < B < 1000$：混合瓶颈，平衡优化
3. $B > 1000$：趋向compute-bound，优化计算

### 2.4.4 KV Cache的影响

考虑KV Cache后的内存需求：
$$Mem_{total} = Mem_{weights} + B \times s \times d \times L \times 2 \times sizeof(float)$$

这会降低有效批大小上限：
$$B_{max} = \frac{Mem_{available} - Mem_{weights}}{s \times d \times L \times 8}$$

## 本章小结

1. **Roofline模型**提供了性能分析的系统框架，通过计算强度判断性能瓶颈
2. **LLM推理特性**：
   - Prefill阶段计算强度较高，易达到compute-bound
   - Decode阶段计算强度极低，通常是memory-bound
3. **关键判则**：
   - Attention计算量在$s > d$时不可忽略
   - 批大小$B > 12d/C$时decode可能转为compute-bound
4. **优化方向**：
   - 小批量：优化内存访问模式
   - 大批量：优化计算效率
   - 长序列：重点优化Attention

## 练习题

### 基础题

1. **计算强度计算**
   给定一个矩阵乘法$C = A \times B$，其中$A$为$m \times k$，$B$为$k \times n$，计算其计算强度。
   *Hint: 考虑FLOPs和需要加载的数据量*

2. **Roofline分析**
   某GPU峰值性能100 TFLOPs，内存带宽1 TB/s，判断计算强度为50的算法是memory-bound还是compute-bound？
   *Hint: 计算$AI_{critical}$*

3. **Attention FLOPs计算**
   对于序列长度1024，隐藏维度4096的self-attention，计算总FLOPs。
   *Hint: 使用本章给出的公式*

4. **批处理效果**
   如果单个请求的计算强度是0.25，需要多大的批大小才能使计算强度达到10？
   *Hint: 假设权重可以完全复用*

### 挑战题

5. **混合精度的影响**
   推导INT8量化对计算强度的影响，假设计算用INT8但内存传输仍是FP16。
   *Hint: 考虑数据类型转换的开销*

6. **多级内存层次**
   考虑L1/L2/L3缓存的Roofline模型，如何修正计算强度的定义？
   *Hint: 不同级别的内存有不同的带宽*

7. **动态批处理策略**
   设计一个算法，根据当前请求队列长度动态选择批大小以最大化吞吐量。
   *Hint: 考虑延迟约束和内存限制*

8. **开放思考：异构计算**
   在CPU+GPU异构系统中，如何分配Attention和FFN的计算以达到最优性能？
   *Hint: 考虑两种硬件的Roofline特性差异*

<details>
<summary>答案（点击展开）</summary>

1. AI = 2mnk / (mn + nk + mk) × sizeof(float)，当矩阵较大时约为k/2

2. AI_critical = 100，算法AI=50 < 100，因此是memory-bound

3. FLOPs = 4×1024×4096² + 2×1024²×4096 ≈ 76.5 GFLOPs

4. B ≈ 40（假设线性增长且忽略激活内存）

5. INT8计算强度约为FP16的2倍（计算量相同但数据量减半）

6. AI_effective = FLOPs / Σ(Data_from_level_i / BW_i)

7. B_opt = min(B_max_memory, max(B_min_latency, B_critical))

8. 将compute-bound的FFN放GPU，memory-bound的Attention放CPU（如果CPU内存带宽更高）

</details>