（交流可以用英文，本文档中文，保留这句）

# 项目说明

## 项目目标
编写一份边缘侧大语言模型和VLM推理加速从software到算法教程markdown，要包含大量的实例（能直接数学算就数学算，或者文字说明，不写代码,但会大量讨论软件包如 vllm，sglang）
算法部分要深入，比如量化历史可以覆盖BinaryConnect/Xor-net/dorefa-net, fq-vit，现状要到 gptq, awq, smoothquant, QuaRot, dfrot，软件要 bitsandbytes
大类包含量化、model pruning, shared-weight merging, 2-4 sparsity, Slimmable Neural Networks
包含边缘侧通用编译如 tensorrt, OpenPPL/PPQ
组织为 index.md + chapter1.md + ...

## 章节结构要求

每个章节应包含：
1. **开篇段落**：简要介绍本章内容和学习目标
2. **本章小结**：总结关键概念和公式
3. **练习题**：
   - 每章包含6-8道练习题
   - 50%基础题（帮助熟悉材料）
   - 50%挑战题（包括开放性思考题）
   - 每题提供提示（Hint）
   - 答案默认折叠，不包含代码
