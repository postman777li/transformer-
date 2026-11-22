# 模块 01｜Transformer 基础原理与整体架构

**状态：已整理；Q001–Q024，共 24 题。**

整理自原聊天《规划Transformer面试题》。保留原有标准回答、必要推导、追问、来源说明及查漏总结。S/A 表示备考优先级，不是频次统计。个人面经为作者自述，二手汇总不等同于原始面经。

**来源迁移说明：** 原聊天部分引用仅有内部检索标记，无法从聊天导出目标 URL，已标记“原链接待补核”；本次为内容迁移，未重新联网核验原面经。已有可导出网址予以保留。追问未单独注明面经依据时，按知识延伸理解。

权威参考：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)。

这次检索中，比较有价值的公开资料包括：

- [字节跳动大模型一面面经（2025-04-02）](https://www.nowcoder.com/feed/main/detail/75f2cc6c30374514bfdafa175b12696c)：直接记录了 Transformer 整体结构、Decoder-only 架构等问题。
- [腾讯 AI Lab NLP 算法岗面经汇总（2025-04-14）](https://www.nowcoder.com/feed/main/detail/75f7d2b159184bc787f658a2da5945e5)：涉及 Transformer 与 LSTM 的比较、Encoder/Decoder 的输入输出、Transformer 复杂度等。
- [百度 AI Infra 提前批一面记录](https://www.nowcoder.com/feed/main/detail/09c161adbf1f459f8e2799d908321420)：涉及整体架构、Attention、FFN、模块参数量等。
- [牛客 Transformer 专题面试题整理](https://www.nowcoder.com/discuss/473903838680875008)：涉及 Encoder–Decoder 交互、并行训练、Decoder Mask 等。

其中，个人发布的面经属于自述资料，无法独立验证其面试实际发生情况；面经汇总和题库则属于二手资料。因此，本题库把“公开面经明确记载”与“为了完整理解而补充的追问”严格区分，不会把它们混成面试频次统计。

## 第一批题目索引

| 题号 | 面试题目 | 优先级 | 来源类型 |
|--------|--------|--------|--------|
| Q001 | 请完整介绍 Transformer 的整体架构 | S | 面经明确记录 |
| Q002 | Transformer 相比 RNN/LSTM 有哪些优势？ | S | 面经汇总记录 |
| Q003 | Transformer Encoder 和 Decoder 有什么区别？ | S | 面经汇总记录 |
| Q004 | Encoder 和 Decoder 之间如何交互？ | A | 专题题库记录 |
| Q005 | 为什么现在主流生成式 LLM 大多采用 Decoder-only？ | S | 多份面经记录 |
| Q006 | Transformer 为什么能够并行训练？Decoder 可以并行吗？ | S | 专题题库记录 |
| Q007 | Transformer 没有循环结构，如何感知 Token 顺序？ | S | 面经汇总记录 |
| Q008 | Transformer 的核心优点和局限是什么？ | A | 综合补充题 |
| Q009 | Transformer 的时间复杂度和空间复杂度是多少？ | S | 面经明确记录 |

S、A 是本题库的备考优先级，而不是经过大规模统计得到的精确出现频率。

---

## Q001｜请完整介绍 Transformer 的整体架构

(S · 核心必背) 架构题 · 建议口述 1–2 分钟

**面经依据：** 字节跳动大模型一面、百度 AI Infra 一面等公开记录均出现了整体架构问题。（原聊天内部引用，原链接待补核）

### ① 面试标准回答

Transformer 是 2017 年 Google 团队在《Attention Is All You Need》中提出的模型架构，核心思想是**使用 Attention 机制建模序列依赖，而不依赖 RNN 的循环计算**。

原版 Transformer 采用 Encoder–Decoder 结构。

Encoder 负责对输入序列进行编码，由多个相同的 Encoder Block 堆叠组成。每个 Block 主要包含 Multi-Head Self-Attention、Feed-Forward Network，以及残差连接和 LayerNorm。

Decoder 负责自回归生成输出。相比 Encoder，它增加了 Masked Self-Attention 和 Cross-Attention。前者确保预测当前位置时不能看到未来 Token，后者让 Decoder 能够读取 Encoder 的输出信息。

此外，Transformer 还利用 Token Embedding 和位置编码表示词语及其顺序。

它的主要优势是**训练阶段可以在序列维度进行并行计算，并通过 Attention 直接建立不同位置之间的联系**。

### ② 深入理解：完整数据流
[原论文架构图：参见 Attention Is All You Need，Figure 1](https://arxiv.org/abs/1706.03762)

原版 Transformer 的 Encoder–Decoder 架构示意。重点看左侧 Encoder、右侧 Decoder，以及二者之间的 Cross-Attention 连接。

假设任务是机器翻译：

输入：`I love you`
输出：`我爱你`

Encoder 处理过程：

```text
输入 Token IDs
      ↓
Token Embedding + Positional Encoding
      ↓
Multi-Head Self-Attention
      ↓
Add & Norm
      ↓
Feed-Forward Network
      ↓
Add & Norm
      ↓
重复 N 层
      ↓
Encoder Hidden States
```

Decoder 的处理过程：

```text
已生成的目标 Token
      ↓
Embedding + Position
      ↓
Masked Self-Attention
      ↓
Cross-Attention ← Encoder 输出
      ↓
Feed-Forward Network
      ↓
Linear → Softmax
      ↓
预测下一个 Token
```

以上是原版 Post-LN Transformer 的简化路径。现代 LLM 可能使用 Pre-Norm、RMSNorm 等不同结构。

### ③ 面试官可能追问

**追问 1：Transformer 的核心是不是只有 Attention？**

不是。Attention 负责 Token 之间的信息交互，FFN 负责逐位置非线性特征变换，残差连接和归一化帮助深层模型稳定训练，位置机制提供顺序信息。Transformer 是这些模块协同工作的结果。

**追问 2：一个 Encoder Block 和一个 Decoder Block 各有几个核心子层？**

原版 Encoder Block 有两个：Self-Attention、FFN。原版 Decoder Block 有三个：Masked Self-Attention、Cross-Attention、FFN。每个子层都配有残差连接和 LayerNorm。

**追问 3：原版 Transformer 为什么采用 Encoder–Decoder？**

因为最初主要面向机器翻译等 Sequence-to-Sequence 任务，需要一部分网络理解源序列，另一部分网络根据源序列逐步生成目标序列。

**追问 4：现在 Llama、GPT 还是这个完整结构吗？**

不是。它们采用 Decoder-only 架构，通常没有独立 Encoder，也没有原版 Encoder–Decoder 之间的 Cross-Attention 子层。

### ④ 代码与笔记对应

[你的 Day 09：原版 Transformer 架构](https://github.com/postman777li/transformer-learning/blob/main/day09/notes.md)

**背诵提醒：** 一定要区分“2017 年原版 Transformer”与“现代 Decoder-only LLM”。这是最容易在介绍架构时说混的地方。

---

## Q002｜Transformer 相比 RNN/LSTM 有哪些优势？为什么会逐渐取代它们成为主流？

(S · 核心必背) 对比题 · 建议口述 40–60 秒

**面经依据：** 腾讯 AI Lab NLP 算法岗公开汇总中，明确记录了 Transformer 与 LSTM 的区别及优势。（原聊天内部引用，原链接待补核）

### ① 面试标准回答

Transformer 相比 RNN/LSTM，主要有三个优势。

第一，**并行计算能力更强**。RNN 在时间维度上依赖前一个 Hidden State，因此需要顺序计算；Transformer 的 Self-Attention 在训练时可以同时处理序列中的多个位置，更适合 GPU 并行。

第二，**更容易建模长距离依赖**。RNN 中相隔较远的 Token 需要经过多步状态传递，容易出现信息衰减；Self-Attention 可以直接建立任意两个位置之间的联系。

第三，**更适合大规模训练和扩展**。Transformer 主要使用矩阵乘法，便于利用 GPU/TPU 的高吞吐计算能力。

但 Transformer 也有局限，例如标准 Self-Attention 的注意力矩阵随序列长度平方增长，长上下文会带来较大计算和显存开销。

### ② 深入理解

RNN：顺序依赖

Transformer：直接交互

设序列有 $T$ 个 Token。

- RNN：$h_t=f(x_t,h_{t-1})$，计算 $h_t$ 依赖 $h_{t-1}$。
- Transformer：在某个 Self-Attention 层内，各个 Token 的 Q、K、V 可以一次通过矩阵运算生成，注意力权重也可批量计算。

注意：这里的“直接交互”指同一层 Attention 中任意两个位置的路径长度短，不意味着模型完全没有顺序约束，也不意味着 Attention 的总计算量更低。

### ③ 面试官可能追问

**追问 1：LSTM 已经解决梯度消失问题，为什么还需要 Transformer？**

LSTM 通过门控机制缓解长距离依赖和梯度问题，但它依然具有时间步上的递归依赖，难以充分并行。Transformer 更大的优势是计算结构和可扩展性，而不是简单地说“LSTM 完全无法记住长距离信息”。

**追问 2：Transformer 在所有方面都优于 RNN 吗？**

不是。标准全注意力在长序列上存在二次复杂度问题，而且原版 Transformer 并不天然适合以固定大小状态逐步处理无限长输入。RNN 或状态空间模型在部分流式、低资源、长序列任务中仍可能有优势。

**追问 3：Transformer 完全没有梯度消失问题吗？**

不是。它通常通过残差连接、归一化和合适的初始化改善深层优化，但深层网络仍可能出现梯度不稳定问题。

### ④ 对应仓库

[Day 01：Transformer 为什么存在](https://github.com/postman777li/transformer-learning/blob/main/day01/notes.md)

---

## Q003｜Transformer 的 Encoder 和 Decoder 有什么区别？

(S · 核心必背) 结构对比题

**面经依据：** 腾讯 AI Lab NLP 岗公开面经汇总明确提及 Encoder 和 Decoder 的输入输出及结构。（原聊天内部引用，原链接待补核）

### ① 面试标准回答

Encoder 和 Decoder 的核心区别在于**注意力的可见范围、网络结构以及承担的任务**。

Encoder 使用双向 Self-Attention，每个 Token 通常可以关注整个输入序列中的其他有效 Token，主要用于构建输入文本的上下文表示。

原版 Decoder 使用 Causal Masked Self-Attention，每个位置只能关注自己和之前的位置，不能看到未来 Token。此外，原版 Decoder 还有 Cross-Attention，用于读取 Encoder 的输出。

因此，Encoder 更偏向对已有输入进行表示建模，而 Decoder 更偏向条件生成或自回归生成。

现代 Decoder-only 模型通常移除了独立的 Encoder 和相应的 Cross-Attention 层，使用一组带因果掩码的 Transformer Blocks 完成整个语言建模过程。

### ② 重点对比表

| 维度 | Encoder | 原版 Decoder |
|--------|--------|--------|
| Self-Attention | 双向 | 因果单向 |
| 能否看未来 Token | 可以 | 不可以 |
| Cross-Attention | 通常没有 | 有 |
| 主要输入 | 源序列 | 目标序列前缀 |
| 主要输出 | 上下文表示 | 目标序列表示及词表预测 |
| 典型模型 | BERT | GPT（Decoder-only 变体） |

这里的“双向”不代表 Encoder 完全没有 Mask，它仍可使用 Padding Mask，防止关注无效填充位置。

### ③ 面试官可能追问

**追问 1：Encoder 和 Decoder 的 Self-Attention 数学公式一样吗？**

核心计算公式一样，都是：

$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$

主要区别在于 Mask。Encoder 通常允许双向关注，Decoder 则通过因果 Mask 禁止访问未来位置。

**追问 2：Decoder 的 Self-Attention 和 Cross-Attention 有什么区别？**

Self-Attention 的 Q、K、V 来自同一序列对应的隐藏状态；Cross-Attention 中，Q 通常来自 Decoder，K 和 V 来自 Encoder 输出。

**追问 3：Decoder-only 和原版 Transformer Decoder 完全一样吗？**

不一样。Decoder-only 模型通常保留带因果掩码的 Self-Attention 和 FFN，但没有原版用于连接独立 Encoder 的 Cross-Attention。

### ④ 对应仓库

[Day 09：Encoder–Decoder 结构](https://github.com/postman777li/transformer-learning/blob/main/day09/notes.md)

---

## Q004｜Transformer 的 Encoder 和 Decoder 之间如何交互？

(A · 重点掌握) 机制题 · Cross-Attention

**来源：** 牛客 Transformer 专题题库明确列有该问题。它是整理题库中的考点，不能直接当作某家公司的独立面经记录。（原聊天内部引用，原链接待补核）

### ① 面试标准回答

原版 Transformer 的 Encoder 和 Decoder 通过 **Cross-Attention（交叉注意力）** 进行交互。

Encoder 首先对输入序列编码，得到一组上下文隐藏状态。

Decoder 在 Cross-Attention 中，使用自身隐藏状态生成 Query，使用 Encoder 的输出生成 Key 和 Value。

这样，Decoder 在生成每个目标 Token 时，就能够根据当前的生成状态，动态关注输入序列中的不同位置，并读取对应的信息。

因此，Cross-Attention 本质上实现了**目标序列对源序列的条件信息读取**。

### ② 深入理解

假设输入：

`I love you`

当前 Decoder 正在预测：

`爱`

则：

- Q：表示 Decoder 当前需要什么信息。
- K：表示 Encoder 中有哪些可供匹配的信息。
- V：表示 Encoder 中能够提供的具体内容。

公式为：

$
Q=H_dW_Q
$

$
K=H_eW_K,\quad V=H_eW_V
$

$
\operatorname{CrossAttn}
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$

其中 $H_d$ 是 Decoder 当前子层输入，$H_e$ 是 Encoder 输出。这里只省略了 Padding Mask 和多头拼接等细节。

### ③ 面试官可能追问

**追问 1：为什么不直接把 Encoder 输出加到 Decoder 的隐藏状态上？**

直接相加需要合适的长度及维度对齐，并且难以让不同目标位置动态选择不同的源序列信息。Cross-Attention 可以为每个 Query 计算针对源序列的不同权重，更灵活地读取信息。

**追问 2：Cross-Attention 中的 Q、K、V 必须来自三个不同的模型吗？**

不需要。它们通常是对两组隐藏状态进行不同线性投影得到的：Q 来自 Decoder 一侧，K 和 V 来自 Encoder 一侧。

**追问 3：Cross-Attention 的注意力矩阵是什么形状？**

单个 Attention Head 下，假设目标序列长度为 $T$，源序列长度为 $S$，则：

$
QK^\top\in\mathbb{R}^{T\times S}
$

如果包含 Batch 和多个 Head，形状通常为：

$
[B,H,T,S]
$

### ④ 对应仓库

[Day 09：Encoder–Decoder 代码](https://github.com/postman777li/transformer-learning/blob/main/day09/code.py)

---

## Q005｜为什么现在的主流生成式大模型大多采用 Decoder-only 架构？

(S · 核心必背) 架构选择题 · 重点考察理解深度

**面经依据：** 字节跳动大模型一面、腾讯 AI Lab NLP 岗面经汇总均记录了这一问题。（原聊天内部引用，原链接待补核）

### ① 面试标准回答

主流生成式 LLM 采用 Decoder-only 架构，主要与**自回归语言建模目标、训练方式、模型扩展能力和推理机制**有关。

第一，Decoder-only 天然适合 Next Token Prediction，即根据前面的 Token 预测下一个 Token，与文本生成任务高度一致。

第二，它使用统一的因果语言建模目标，可以利用大量无标注文本进行预训练，不需要为每条训练数据准备独立的输入输出配对。

第三，架构比较统一，便于扩大模型参数量、训练数据量和计算规模。训练时可以借助 Causal Mask 对序列中各个位置的预测进行并行计算。

第四，自回归推理能够使用 KV Cache，复用历史 Token 的 K、V，减少重复计算。

因此，Decoder-only 在开放式文本生成场景下形成了非常成熟的训练与推理生态。

但这不意味着 Encoder-only 或 Encoder–Decoder 已经没有价值，它们在文本理解、分类、机器翻译等任务中仍然适用。

### ② 深入理解：三种架构的任务区别

Encoder-only

双向理解

Decoder-only

因果预测

Encoder–Decoder



编码后生成

| 架构 | 代表模型 | 常见任务 |
|--------|--------|--------|
| Encoder-only | BERT | 分类、匹配、表征学习 |
| Decoder-only | GPT、Llama、Qwen 的生成式模型 | 自回归文本生成 |
| Encoder–Decoder | T5、BART | 翻译、摘要、序列转换 |

注意：这些是典型应用而非绝对限制。Decoder-only 也能够完成分类任务，Encoder–Decoder 同样能够进行开放式生成。

### ③ 面试官可能追问

**追问 1：Decoder-only 为什么也能理解文本？**

虽然它使用因果注意力，但随着网络层数增加，后面的 Token 可以不断聚合之前 Token 的信息。经过大规模预训练后，模型能够学习语义、句法、知识和推理相关的模式，因此既可以建模输入，也可以生成输出。

**追问 2：Encoder–Decoder 不能使用 KV Cache 吗？**

可以。Encoder–Decoder 推理时可以缓存 Decoder 的历史 Self-Attention K/V，并缓存由 Encoder 输出计算得到的 Cross-Attention K/V。

所以不能简单地说“Decoder-only 流行是因为其他架构不能做 KV Cache”。

**追问 3：Decoder-only 是否一定比 Encoder–Decoder 训练效率高？**

不一定。训练效率取决于任务、输入输出长度、模型大小、实现方式和硬件。Decoder-only 的优势主要体现在统一的自回归目标及其规模化训练生态，而非所有设置下都具有更低计算量。

**追问 4：为什么 BERT 不适合直接进行标准自回归生成？**

BERT 的常见预训练方式使用双向上下文和 Masked Language Modeling，并不是原生的从左到右 Next Token Prediction。要用于自回归生成，需要调整注意力约束或训练目标。

### ④ 对应仓库

[Day 10：GPT 与 Decoder-only](https://github.com/postman777li/transformer-learning/blob/main/day10/notes.md)

---

## Q006｜Transformer 为什么能够并行训练？Decoder 可以并行吗？

(S · 核心必背) 训练机制题 · 容易被连续追问

**来源：** 牛客 Transformer 专题题库将该问题列为单独考点。（原聊天内部引用，原链接待补核）

### ① 面试标准回答

Transformer 的并行性主要体现在：**训练时同一层中不同 Token 位置的计算可以通过矩阵运算同时进行，而不必像 RNN 一样按时间步顺序更新隐藏状态。**

Encoder 使用 Self-Attention，可以同时计算整个输入序列的 Q、K、V 和注意力输出。

Decoder 虽然使用 Causal Mask，但在训练阶段，因为整个目标序列已经给定，可以通过 Teacher Forcing 一次输入所有目标 Token，并同时计算各位置的 Next Token Prediction Loss。

Causal Mask 只限制每个位置能访问哪些信息，不要求计算过程必须逐 Token 执行。

但在自回归推理阶段，Decoder 通常必须先生成当前 Token，才能确定下一步输入，因此不同生成步骤之间存在顺序依赖。

所以，**Decoder 训练阶段可以沿序列维度并行，而常规自回归生成不能在时间步之间完全并行。**

### ② 深入理解：训练和推理的区别

假设训练文本为：

`我 喜欢 学习 AI`

训练时：

| 输入位置 | 预测目标 |
|--------|--------|
| 我 | 喜欢 |
| 我 喜欢 | 学习 |
| 我 喜欢 学习 | AI |

实际实现可以把移位后的输入和标签构成张量，通过一次模型前向传播计算全部位置的预测结果。

例如，假设 `input_ids` 形状为 `[B,T]`，则：

```python
logits = model(input_ids)
# [B, T, vocab_size]
```

每个位置都有一个预测分布。通过 Causal Mask 保证第 $t$ 个位置不能读取之后的 Token，就可以同时计算各位置的损失。

但是推理时：

```text
输入 Prompt
    ↓
生成 Token 1
    ↓
生成 Token 2
    ↓
生成 Token 3
```

后一步通常依赖前一步生成的具体结果。

### ③ 面试官可能追问

**追问 1：既然 Decoder 有 Mask，为什么训练不需要一个 Token 一个 Token 地算？**

因为 Mask 只是让未来位置在当前 Query 的 Attention 中不可见。整个 Q、K、V 张量仍可以一次生成，并通过矩阵运算计算所有允许的注意力关系。

**追问 2：Transformer 所有层都可以并行执行吗？**

不是。同一层内的很多计算可以并行，但第 $l+1$ 层需要第 $l$ 层的输出，因此网络层之间仍存在依赖。

**追问 3：推理阶段完全不能并行吗？**

也不是。Prefill 可以并行处理已有 Prompt 的多个位置；不同请求可以批处理；同一步内部的矩阵计算也可并行。投机解码等技术还可以尝试提高多 Token 生成效率。但普通自回归解码的 Token 步骤存在因果依赖。

**追问 4：Teacher Forcing 是什么？**

训练时使用真实目标序列中前面的 Token 作为模型输入，而不是每一步都使用模型自己上一步生成的结果。这使得整段目标序列可以进行并行训练。

### ④ 对应仓库

[Day 12：Transformer 训练流程](https://github.com/postman777li/transformer-learning/blob/main/day12/notes.md)

---

## Q007｜Transformer 没有 RNN 的循环结构，它是如何感知 Token 顺序的？

(S · 核心必背) 位置机制题

**面经依据：** 腾讯 AI Lab NLP 岗面经汇总中出现了 Transformer 如何体现时序信息，以及 BERT 为什么使用正弦余弦位置编码等问题。（原聊天内部引用，原链接待补核）

### ① 面试标准回答

标准 Self-Attention 本身没有明确的位置顺序约束。如果不额外引入位置信息，它无法仅通过注意力机制区分序列中 Token 的不同排列。

因此，Transformer 需要引入位置机制，使模型同时获得 Token 的语义信息和顺序信息。

原版 Transformer 使用 Sinusoidal Positional Encoding，通过不同频率的正弦和余弦函数为不同位置构造向量，再与 Token Embedding 相加。

后续模型也使用过可学习绝对位置编码、相对位置编码、RoPE、ALiBi 等方案。

其中，现代许多生成式 LLM 使用 RoPE，将位置信息以旋转变换的方式引入 Q、K，使注意力分数能够体现相对位置关系。

### ② 深入理解

原版 Transformer 的输入为：

$
X=\operatorname{Embedding}(tokens)+PE
$

例如：

`我 喜欢 你` 与 `你 喜欢 我`

即使包含相同的 Token，位置编码也使不同位置上的输入表示有所区别。

原版正弦余弦编码公式：

$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d}}\right)
$

$
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d}}\right)
$

这里的 $pos$ 表示 Token 位置，$i$ 表示位置编码的维度索引，$d$ 表示模型的隐藏维度。

### ③ 面试官可能追问

**追问 1：为什么不加位置编码，Self-Attention 就不能正确感知顺序？**

不考虑 Mask 等额外位置机制时，Self-Attention 对输入 Token 的排列具有置换等变性：如果同时打乱 Token 的排列，输出也会相应打乱，而不会自动知道原来的先后关系。

**追问 2：为什么使用正弦余弦函数？**

它提供了确定性且连续变化的位置表示，而且不同频率的组合能够表达不同尺度的位置模式。相对位置偏移也能通过三角函数关系表示为线性变换，这是这种设计的一个重要数学性质。

**追问 3：Causal Mask 不也包含顺序信息吗？**

包含。因果 Mask 会引入可见范围的先后约束，因此严格来说，带 Causal Mask 的注意力并不完全具有无掩码 Self-Attention 的置换等变性。

但 Mask 只能约束哪些位置可见，不等于完整提供距离、相对位置等信息，因此现代 Decoder-only 模型通常仍使用额外的位置机制。

**追问 4：RoPE 与绝对位置编码的主要区别是什么？**

绝对位置编码通常把位置表示加入隐藏状态；RoPE 则通过旋转变换作用于 Q、K，使其点积自然体现相对位置差异。

这一问题将在模块 4 深入推导。

### ④ 对应仓库

[Day 07：Position 和 Mask](https://github.com/postman777li/transformer-learning/blob/main/day07/notes.md)

---

## Q008｜Transformer 的核心优势和局限性分别是什么？

(A · 重点掌握) 系统理解题 · 综合补充

**来源说明：** 本题是结合原始 Transformer 论文、面经中对架构优缺点的追问方向整理的综合补充题，并非目前已确认的某一场面试的原题。（原聊天内部引用，原链接待补核）

### ① 面试标准回答

Transformer 的优势主要包括三点。

第一，具有较强的并行计算能力。相比循环神经网络，Transformer 在训练阶段能够同时计算多个 Token 的表示，适合 GPU 加速。

第二，能够直接建立长距离 Token 之间的依赖关系，避免信息必须通过许多连续时间步传递。

第三，具有较好的可扩展性，能够通过增加模型规模、数据规模和计算资源提升模型能力，因此成为现代大语言模型的重要基础。

它的局限主要有三点。

第一，标准全注意力的计算量和注意力矩阵存储需求都包含序列长度的平方项，长上下文成本较高。

第二，自回归生成存在 Token 之间的顺序依赖，难以一次性并行生成所有未来 Token。

第三，普通注意力本身不具有 RNN 那样的递归顺序归纳偏置，需要通过位置编码、因果掩码等机制表示顺序信息。

因此，近年来的大模型研究有很多工作围绕长上下文、高效 Attention、KV Cache、模型压缩和推理加速展开。

### ② 深入理解：优势与代价

| 优势 | 对应的代价或限制 |
|--------|--------|
| 全局 Token 信息交互 | 标准全注意力有二次序列长度成本 |
| 训练阶段并行性强 | 自回归生成存在时间步依赖 |
| 矩阵计算适合 GPU | 计算吞吐和显存带宽会成为瓶颈 |
| 易于扩大参数规模 | 训练、部署和推理资源成本增加 |
| 架构灵活 | 需要额外的位置、掩码和训练设计 |

### ③ 面试官可能追问

**追问 1：既然 Transformer 能建模长距离依赖，为什么长文本仍然困难？**

能够建立长距离连接不等于能无成本、无损失地利用任意长上下文。长序列会增加计算、显存和 KV Cache 开销，模型还可能出现远距离信息检索不准确、上下文利用不足等问题。

**追问 2：FlashAttention 解决了什么问题？**

FlashAttention 通过 IO-aware 的分块计算和在线 Softmax，减少对 GPU 高带宽显存的读写，避免显式保存完整的注意力分数矩阵，从而提升实际计算效率并降低注意力计算的中间显存开销。

但标准精确 FlashAttention 的算术复杂度仍有关于序列长度的二次项，不能说它直接把全注意力变成了线性计算复杂度。

**追问 3：如果序列极长，可以考虑什么改进？**

可以考虑稀疏或局部 Attention、滑动窗口、上下文压缩、分块处理、线性注意力、状态空间模型，以及结合具体场景优化 KV Cache 管理等。

这些方案通常需要在计算效率、信息覆盖范围和模型质量之间做权衡。

### ④ 对应仓库

[Day 16：FlashAttention、复杂度与长上下文（待完善）](https://github.com/postman777li/transformer-learning/blob/main/day16/notes.md)

---

## Q009｜Transformer 的时间复杂度和空间复杂度是多少？为什么说 Attention 是 O(n²)？

(S · 核心必背) 数学推导题 · 必须能现场推导

**面经依据：** 腾讯 AI Lab NLP 面经汇总明确包含 Transformer 复杂度问题；牛客 Transformer 专题也包含计算量相关考点。（原聊天内部引用，原链接待补核）

### ① 面试标准回答

标准 Transformer 的 Self-Attention 之所以具有关于序列长度的二次复杂度，主要来自计算 $QK^\top$。

设序列长度为 $n$，隐藏维度为 $d$。

Q、K、V 的线性投影复杂度通常为 $O(nd^2)$。

而计算 $QK^\top$ 时，需要建立序列中所有 Token 两两之间的相关性，因此复杂度为 $O(n^2d)$。

随后注意力权重与 V 相乘，也需要 $O(n^2d)$ 的计算量。

另外，FFN 的计算复杂度通常为 $O(nd^2)$，这里假设 FFN 中间维度与 $d$ 成固定比例。

所以一个标准 Transformer Block 的主要计算复杂度为：

$
\boxed{O(nd^2+n^2d)}
$

如果重点关注注意力的二次序列长度项，可以说 Self-Attention 包含 $O(n^2d)$ 的计算，而显式存储注意力矩阵通常需要 $O(Hn^2)$ 的空间，其中 $H$ 为注意力头数。

但不能笼统地说 Transformer 在任何情况下的总复杂度都只是 $O(n^2)$。

### ② 深入推导

设：

- $n$：序列长度
- $d$：模型隐藏维度
- $H$：注意力头数
- $d_h=d/H$：每个 Head 的维度

一个 Attention Head 中：

$
Q\in\mathbb{R}^{n\times d_h},
\quad
K\in\mathbb{R}^{n\times d_h}
$

计算：

$
QK^\top
$

输出维度：

$
[n,d_h]\times[d_h,n]=[n,n]
$

其计算量为：

$
O(n^2d_h)
$

所有 $H$ 个 Head 合计：

$
H\cdot O(n^2d_h)=O(n^2d)
$

由于注意力权重还需要乘 V，因此注意力核心计算存在两个同阶的矩阵乘法。

### ③ 为什么长上下文成本增长很快？
（长度翻倍时，注意力矩阵元素数量变为四倍。）

当序列长度增加 2 倍时：

- Q、K、V 的 Token 数量增加 2 倍。
- 显式注意力矩阵的元素数量增加 4 倍。
- 标准全注意力核心的 $n^2d$ 计算项也增加约 4 倍（固定 $d$）。

这正是长上下文优化如此重要的原因。

### ④ 面试官可能追问

**追问 1：既然 Attention 是二次复杂度，为什么有时 FFN 的计算量反而更大？**

因为两者不仅取决于序列长度，还取决于隐藏维度。

Attention 的主要计算包含 $n^2d$ 项，而 FFN 主要包含 $nd^2$ 项。

当 $d$ 较大、$n$ 相对较小时，FFN 和线性投影的计算量可能非常显著，甚至超过注意力核心计算。

**追问 2：Attention 头数从 8 增加到 16，复杂度一定翻倍吗？**

不一定。

如果模型总隐藏维度 $d$ 固定，每个 Head 的维度由 $d/8$ 减小到 $d/16$，那么标准 MHA 的主导矩阵乘法计算量在渐近意义上基本不变。

但实际运行时间可能受到 Kernel 实现、并行度、访存等因素影响。

**追问 3：Attention 的空间复杂度就是 $O(n^2)$ 吗？**

要区分存储对象。

显式注意力矩阵在单层多头情况下占用 $O(Hn^2)$ 的元素空间。

但模型还有 Q、K、V、隐藏状态、参数以及反向传播需要保存的中间数据，因此完整模型的显存使用不能只用这一项描述。

另外，FlashAttention 等实现可以避免显式保存完整注意力矩阵，显著减少中间显存需求。

**追问 4：KV Cache 的空间复杂度也是 $O(n^2)$ 吗？**

不是。对于标准多头注意力，KV Cache 的容量与已缓存 Token 数量通常呈线性关系。

其元素数量可以写为：

$
\boxed{2BLnHd_h}
$

其中 $B$ 是 Batch Size，$L$ 是层数，$n$ 是缓存长度，$H$ 是 KV Head 数量，$d_h$ 是每个 KV Head 的维度。

这和训练或 Prefill 时可能出现的 $n\times n$ 注意力矩阵不是同一回事。

### ⑤ 对应仓库

[Day 05：Scaled Dot-Product Attention](https://github.com/postman777li/transformer-learning/blob/main/day05/notes.md)

[Day 14：KV Cache 与推理机制](https://github.com/postman777li/transformer-learning/blob/main/day14/notes.md)

---



## Q010｜一个 Prompt 输入 Decoder-only 大模型后，是如何一步步生成 Token 的？

(S · 必背) 公开面经汇总记录

**① 面试标准回答**

整个过程分为六步：

1. **Tokenization**：将输入文本转成 Token IDs。
2. **Embedding**：把 Token IDs 映射为连续向量，并引入位置信息。
3. **Transformer Blocks**：经过多层 Attention、FFN、残差及归一化，得到上下文隐藏表示。
4. **LM Head**：将最后一层的隐藏状态映射为词表维度的 Logits。
5. **Decoding**：对 Logits 进行必要处理，通过 Greedy、Temperature、Top-k、Top-p 等策略选择下一个 Token。
6. **Autoregressive Generation**：将新 Token 加入上下文，重复生成，直到 EOS 或其他停止条件。

实际高效推理一般分为 Prefill 和 Decode 两个阶段，并使用 KV Cache 减少重复计算。

**② 追问**

**Q：为什么最后只取最后一个位置的 Logits？**

常规自回归推理中，每个位置的 Logits 对应其下一个 Token 的预测。当前需要生成的是整个已知前缀之后的新 Token，因此使用最后一个有效位置的 Logits。

**Q：Logits 就是概率吗？**

不是。Logits 是未经归一化的实数分数，经过 Softmax 才能得到概率分布。

**③ 来源与仓库**

公开面经汇总：《2025–2026 年 AI 算法与模型研发大厂面试高频问题》。（原聊天内部引用，原链接待补核）

仓库：[Day 11：Mini-GPT](https://github.com/postman777li/transformer-learning/blob/main/day11/notes.md)、[Day 13：文本生成](https://github.com/postman777li/transformer-learning/blob/main/day13/notes.md)。

---

## Q011｜Transformer 为什么先进行 Token Embedding？原版为什么还要乘 $\sqrt{d_{\text{model}}}$？

(A · 重点) 公开题库收录

**① 面试标准回答**

神经网络无法直接处理离散的 Token ID，因此需要 Embedding 层将每个 Token 映射为可学习的稠密向量。

假设词表大小为 $V$，隐藏维度为 $d$，Embedding 本质上是一张可学习的矩阵：

$
E\in\mathbb{R}^{V\times d}
$

输入 Token ID 后查表，得到对应的 $d$ 维向量。

原版 Transformer 还将 Embedding 乘以 $\sqrt{d_{\text{model}}}$，主要是调整其数值尺度，使 Token Embedding 与位置编码相加时具有合适的相对量级。

**注意：** 这种缩放并非所有现代 LLM 都采用，且不能把它解释为保证两者方差严格相等。

**② 追问**

**Q：Embedding 和 One-hot 有什么区别？**

One-hot 是固定的高维稀疏表示；Embedding 是通过训练得到的低维稠密向量，能够学习 Token 间的统计与语义关系。

**Q：为什么 Embedding 层本质上也是一个可训练矩阵？**

每个 Token ID 对应矩阵中的一行，训练时梯度会更新相关向量，因此模型能够学习适合任务的表示。

**③ 来源与仓库**

GitHub《史上最全 Transformer 面试题》明确收录了 Embedding 缩放问题。（原聊天内部引用，原链接待补核）

仓库：[Day 02：Embedding](https://github.com/postman777li/transformer-learning/blob/main/day02/notes.md)。

---

## Q012｜为什么大模型的 Embedding 和 LM Head 可以共享权重？有什么好处？

(S · 必背) 真实面经记录

**① 面试标准回答**

Embedding 层负责将 Token ID 映射成隐藏向量，LM Head 则将隐藏向量映射回词表维度的预测分数。

假设：

$
E\in\mathbb{R}^{V\times d}
$

输入查表使用 $E$，输出投影可以使用：

$
\mathrm{Logits}=HE^\top
$

两者共享同一组词表表示参数，这种方式称为 **Weight Tying（权重绑定）**。

主要好处是减少参数量，尤其当词表很大时，可以节省约 $Vd$ 个参数，同时为输入表示与输出词表评分提供共享的参数基础。

但共享权重也会约束模型的表达自由度，因此不是所有 LLM 都采用这一设计。

**② 追问**

**Q：如果词表大小为 100,000，隐藏维度为 4096，共享权重能节省多少参数？**

$
100000\times4096=409600000
$

约节省 **4.096 亿个参数**，不计偏置等额外参数。

**Q：共享权重是不是把 LM Head 删除了？**

不是。LM Head 的输出投影运算仍然存在，只是复用了 Embedding 的参数。

**③ 来源与仓库**

[联想大模型开发一面面经](https://www.nowcoder.com/discuss/872219056412098560)，明确记录了 Embedding 与 LM Head 权重共享问题。（原聊天内部引用，原链接待补核）

仓库：[Day 11：Mini-GPT 代码](https://github.com/postman777li/transformer-learning/blob/main/day11/code.py)。

---

## Q013｜Transformer Decoder 训练时，为什么目标序列需要右移一位？

(S · 必背) 公开面试专题收录

**① 面试标准回答**

因为自回归语言模型的目标是：**根据前面的 Token 预测下一个 Token**。

训练时，需要让第 $t$ 个位置的输入对应第 $t+1$ 个位置的预测标签，因此输入和标签需要错位对齐。

例如：

```text
原始序列：<BOS> 我 喜欢 AI <EOS>

Decoder输入：<BOS> 我 喜欢 AI
预测标签：   我   喜欢 AI <EOS>
```

这样可以一次前向传播计算多个位置的预测损失。

与此同时，Causal Mask 负责阻止模型从未来位置获取信息，避免信息泄漏。

**② 追问**

**Q：有了右移，为什么还需要 Causal Mask？**

右移只解决输入与标签的对应关系。如果没有 Causal Mask，第一个位置仍可能通过 Self-Attention 读取后续输入 Token，从而提前获取本不应该知道的信息。

**Q：代码里一定要显式右移输入吗？**

不一定。有些训练框架会在模型内部对 Logits 和 Labels 进行移位对齐，因此需要检查具体实现，避免重复移位。

**③ 来源与仓库**

牛客 Transformer 专题讨论了 Decoder 输入及 Shift Label，并有评论专门纠正训练并行的理解。（原聊天内部引用，原链接待补核）

仓库：[Day 12：训练与标签对齐](https://github.com/postman777li/transformer-learning/blob/main/day12/notes.md)。

---

## Q014｜Transformer 中有哪些 Mask？分别解决什么问题？

(S · 必背) 公开题库收录

**① 面试标准回答**

Transformer 中最常见的是两类 Mask：

| 类型 | 作用 |
|--------|--------|
| Padding Mask | 屏蔽 Padding Token，防止无效填充位置参与注意力 |
| Causal Mask | 屏蔽未来位置，防止自回归预测时看到未来 Token |

在 Attention 中，通常在 Softmax 前向需要屏蔽的位置加上负无穷：

$
\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right)
$

这样对应位置经过 Softmax 后权重为零。

在现代 LLM 中，还可能存在滑动窗口 Mask、Prefix-LM Mask、Block Mask 等特殊形式。

**② 追问**

**Q：Mask 为什么放在 Softmax 之前？**

因为 Softmax 会将分数转换为归一化权重。提前将无效位置设为负无穷，才能使这些位置的权重变为零，并让其余有效位置重新归一化。

**Q：Attention Mask 和 Loss Mask 是一回事吗？**

不是。Attention Mask 控制哪些 Token 可以相互关注；Loss Mask 决定哪些位置参与损失计算，例如 Padding 或不需要监督的 Prompt Token 可以不计入 Loss。

**③ 来源与仓库**

GitHub Transformer 面试题库收录 Padding Mask 问题；牛客专题也讨论了 Decoder Mask。（原聊天内部引用，原链接待补核）

仓库：[Day 07：Position 与 Mask](https://github.com/postman777li/transformer-learning/blob/main/day07/notes.md)。

---

## Q015｜为什么 Transformer 最后需要 Linear + Softmax？为什么不能直接用隐藏状态预测 Token？

(A · 重点) 系统补充题

**① 面试标准回答**

Transformer 的隐藏状态通常是 $d$ 维特征向量，而模型需要从大小为 $V$ 的词表中选择下一个 Token，因此需要通过 LM Head 将隐藏维度映射到词表维度。

$
H\in\mathbb{R}^{B\times T\times d}
$

$
\mathrm{Logits}\in\mathbb{R}^{B\times T\times V}
$

每个位置输出 $V$ 个分数，分别表示词表中各个 Token 的未归一化预测分数。

训练时通常直接将 Logits 送入 CrossEntropyLoss；推理时则根据采样策略处理 Logits，必要时通过 Softmax 获得概率分布。

**② 追问**

**Q：为什么训练时不先手动 Softmax？**

PyTorch 的 `CrossEntropyLoss` 内部已经包含 LogSoftmax 与负对数似然损失的计算。直接输入 Logits 更方便，也具有更好的数值稳定性。

**Q：LM Head 一定是复杂神经网络吗？**

不一定。常见实现中就是一个线性投影层。

**③ 来源与仓库**

系统补充题。用于补齐 Prompt 到预测结果的完整数据流，非已核实的独立面经原题。

仓库：[Day 11：Mini-GPT](https://github.com/postman777li/transformer-learning/blob/main/day11/notes.md)。

---

## Q016｜Transformer 为什么还需要 FFN？Attention 不是已经能处理信息了吗？

(S · 必背) 真实面经记录

**① 面试标准回答**

Attention 和 FFN 承担不同的功能。

Attention 主要负责不同 Token 之间的信息交互，让每个位置能够整合上下文。

FFN 则对每个 Token 的隐藏表示进行非线性特征变换，提高模型的表达能力。

原版 FFN 为：

$
\operatorname{FFN}(x)=W_2\operatorname{ReLU}(W_1x+b_1)+b_2
$

通常先升维，再降维。

因此可以简单概括为：

**Attention 负责跨 Token 信息混合，FFN 负责逐 Token 非线性特征处理。**

**② 追问**

**Q：FFN 在不同 Token 之间共享参数吗？**

共享。同一层 FFN 使用相同的权重处理不同位置，但每个位置接收的隐藏状态不同。

**Q：FFN 会直接进行 Token 之间的信息交互吗？**

标准 Position-wise FFN 不会。它独立处理每个位置，跨 Token 的信息交互主要由 Attention 完成。

**③ 来源与仓库**

[雄岸科技大模型算法面经](https://www.nowcoder.com/feed/main/detail/a3373173d2604ce18530a01aa3cc492e)，明确记录了 FFN 相关提问。（原聊天内部引用，原链接待补核）

仓库：[Day 08：FFN](https://github.com/postman777li/transformer-learning/blob/main/day08/notes.md)。

---

## Q017｜Transformer 为什么需要残差连接？如果去掉会怎样？

(S · 必背) 真实面经记录

**① 面试标准回答**

残差连接的核心作用是**改善深层网络的优化过程和梯度传播**。

其基本形式为：

$
y=x+F(x)
$

网络不必每一层都重新学习完整表示，而可以学习在输入基础上的增量修正。

反向传播时，残差路径提供了额外的梯度传递通道，有助于缓解深层网络中的梯度消失和优化困难。

如果去掉残差连接，深层 Transformer 通常会更难训练，训练稳定性及最终效果可能下降。

**② 追问**

**Q：残差连接是否保证梯度不会消失？**

不能保证。它提供直接梯度路径，但实际梯度还受到归一化、权重、网络深度等因素影响。

**Q：为什么残差连接要求输入输出维度一致？**

因为需要进行逐元素相加。如果维度不一致，就需要先通过线性投影等方式对齐。

**③ 来源与仓库**

雄岸科技大模型算法面经明确记录了残差连接的作用。（原聊天内部引用，原链接待补核）

仓库：[Day 08：Residual Connection](https://github.com/postman777li/transformer-learning/blob/main/day08/notes.md)。

---

## Q018｜Transformer 为什么要堆叠多个 Block？各层参数是否共享？

(A · 重点) 系统补充题

**① 面试标准回答**

Transformer 堆叠多个 Block，是为了通过多层信息交互和非线性变换，逐步构建更复杂的上下文表示。

一层 Attention 可以聚合 Token 间的信息，而多层结构允许模型反复处理和组合这些信息，从而提高表达能力。

原版 Transformer 的不同 Block **结构相同，但参数通常不共享**。每层都有独立的 Attention 和 FFN 参数，可以学习不同的特征变换。

需要注意，参数共享也可以作为特定架构设计，例如 ALBERT 就采用了跨层参数共享技术。

**② 追问**

**Q：增加层数一定能提升模型性能吗？**

不一定。更深的网络通常具有更大的表达容量，但也会增加计算开销、优化难度，并可能受到数据和训练配置限制。

**Q：层与层之间能并行计算吗？**

普通顺序堆叠的 Transformer 不行。后一层依赖前一层的输出，但同一层内部有大量并行计算。

**③ 来源与仓库**

系统补充题。结构堆叠属于公开面经经常涉及的整体架构解释内容，但目前没有把本题确认成某家公司的独立原题。

仓库：[Day 11：多层 Transformer Blocks](https://github.com/postman777li/transformer-learning/blob/main/day11/code.py)。

---

## Q019｜Transformer 的 Dropout 有什么作用？训练与推理时有什么区别？

(A · 重点) 公开题库延伸

**① 面试标准回答**

Dropout 是一种正则化方法，训练时随机将部分激活置零，以减少模型对特定神经元组合的过度依赖，改善泛化能力。

原版 Transformer 在多个位置使用 Dropout，包括子层输出、位置编码与 Embedding 相加后的表示，以及注意力权重等。

在 PyTorch 中：

- `model.train()`：启用 Dropout 等训练行为。
- `model.eval()`：关闭 Dropout，使推理输出不再受到随机丢弃的影响。

现代大模型预训练中，也有不少模型将 Dropout 设置为零，因此不能认为所有 Transformer 都必须使用 Dropout。

**② 追问**

**Q：Dropout 会直接把模型参数永久置零吗？**

不会。训练时随机置零的是相应激活值，不是永久删除模型参数。

**Q：`model.eval()` 是否等同于 `torch.no_grad()`？**

不是。`eval()` 改变 Dropout 等模块的行为；`no_grad()` 关闭自动求导记录。推理时通常会同时使用 `eval()` 和 `torch.inference_mode()`。

**③ 来源与仓库**

公开 Transformer 架构题库涉及 FFN 中的 Dropout 及其训练结构；本题的完整问法属于系统延伸。（原聊天内部引用，原链接待补核）

仓库：[Day 12：训练代码](https://github.com/postman777li/transformer-learning/blob/main/day12/code.py)。

---



## Q020｜Transformer 为什么使用 LayerNorm，而不是 BatchNorm？

(S · 必背) 公开面试题库收录

**① 面试标准回答**

Transformer 主要采用 LayerNorm，因为它更适合变长序列建模，并且不依赖 Batch 内其他样本的统计信息。

两者最主要的区别是归一化维度：

- **BatchNorm**：通常针对每个特征或通道，利用 Batch 维度等位置上的统计量进行归一化。
- **LayerNorm**：通常对每个 Token 的隐藏特征维度进行归一化，不依赖其他样本。

在 NLP 中，序列长度可能不同，Batch Size 也可能变化。如果使用 BatchNorm，统计量可能受到 Padding、Batch 组成等因素影响。

因此，LayerNorm 更方便处理序列数据，也具有更一致的训练和推理行为。

**② 高频追问**

**Q：输入是 `[B,T,D]`，LayerNorm 通常对哪个维度归一化？**

通常对最后的隐藏维度 `D` 进行归一化，每个 Token 独立计算均值和方差。

**Q：LayerNorm 训练和推理时使用的统计量一样吗？**

标准 LayerNorm 都基于当前输入计算统计量，不需要像常规 BatchNorm 那样维护用于推理的 Running Mean 和 Running Variance。

**③ 来源**

[牛客｜算法岗常见面试题（八）：Transformer](https://www.nowcoder.com/discuss/473903838680875008?sourceSSR=dynamic)，第 10 题明确收录了 LN 与 BN 的比较。（原聊天内部引用，原链接待补核）

仓库对应：`day08/notes.md`

---

## Q021｜Transformer 中 Pre-Norm 和 Post-Norm 有什么区别？为什么现代 LLM 多采用 Pre-Norm？

(S · 必背) 公开岗位面经记录

**① 面试标准回答**

Pre-Norm 和 Post-Norm 的核心区别是 LayerNorm 与残差连接的位置不同。

Post-Norm：

$
y=\operatorname{LN}(x+F(x))
$

Pre-Norm：

$
y=x+F(\operatorname{LN}(x))
$

原版 Transformer 使用 Post-Norm，而现代许多大语言模型采用 Pre-Norm 或其变体。

主要原因是 Pre-Norm 的残差主路径更加直接，梯度传播通常更稳定，尤其适合训练深层网络。

不过 Pre-Norm 也可能存在表示演化或有效深度方面的权衡，因此不能简单认为它在所有任务中都优于 Post-Norm。

**② 高频追问**

**Q：为什么说 Pre-Norm 更有利于梯度传播？**

因为它具有直接的残差恒等路径 $x\rightarrow y$，反向传播时存在不必经过当前子层 LayerNorm 的梯度路径。

**Q：Llama 使用的是 Pre-Norm 还是 Post-Norm？**

Llama 系列典型结构采用 Pre-Norm 思路，但使用的是 RMSNorm，而不是原版 LayerNorm。

**③ 来源**

[牛客｜26 校招腾讯大模型算法面经](https://www.nowcoder.com/feed/main/detail/a9d3a526d0374d31a4ae6cddabce97ee?urlSource=home-api) 明确出现 Pre-Norm 与 Post-Norm 的设计差异问题。（原聊天内部引用，原链接待补核）

原论文结构参考：[The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)。

仓库对应：`day08/notes.md`、`day15/notes.md`

---

## Q022｜原版 Transformer 有多少层？几个 Attention Head？隐藏维度是多少？

(A · 重点) 原论文参数补充题

**① 面试标准回答**

2017 年原版 Transformer Base 模型的核心配置为：

| 参数 | 数值 |
|--------|--------|
| Encoder 层数 | 6 |
| Decoder 层数 | 6 |
| 模型隐藏维度 $d_{\text{model}}$ | 512 |
| Attention Head 数 | 8 |
| 每个 Head 的 Q/K 维度 | 64 |
| FFN 中间维度 $d_{ff}$ | 2048 |
| Dropout | 0.1 |

其中：

$
d_k=d_v=\frac{512}{8}=64
$

需要注意，这只是**原版 Transformer Base** 的配置，不是 Transformer 架构必须遵守的固定参数。

**② 高频追问**

**Q：为什么每个 Head 是 64 维？**

在原版设计中，总隐藏维度为 512，划分成 8 个 Head，因此每个 Head 使用 64 维。

**Q：多头拼接后为什么仍然是 512 维？**

8 个 Head 每个输出 64 维，拼接后就是 $8\times64=512$ 维，再经过输出线性投影。

**③ 来源**

原论文参数配置，属于系统补充题，不将其冒充为已验证的具体公司面经原题。（原聊天内部引用，原链接待补核）

仓库对应：`day06/notes.md`、`day09/notes.md`

---

## Q023｜原版 Transformer 一共有多少个残差连接？分别位于哪里？

(A · 重点) 公开岗位面经记录

**① 面试标准回答**

原版 Transformer 每个 Encoder Block 有两个残差连接：

1. Multi-Head Self-Attention 子层之后。
2. Feed-Forward Network 子层之后。

每个 Decoder Block 有三个残差连接：

1. Masked Self-Attention 子层之后。
2. Encoder–Decoder Cross-Attention 子层之后。
3. Feed-Forward Network 子层之后。

原版 Encoder 和 Decoder 各有 6 层，因此按这两组堆叠 Block 内的子层计算：

$
6\times2+6\times3=\boxed{30}
$

即整个原版 Base 模型的 Encoder–Decoder 堆叠中共有 30 个子层残差连接。

**② 高频追问**

**Q：为什么不直接说 Transformer 有两个残差连接？**

因为“两个残差连接”指的是一个 Encoder Block。Decoder Block 有三个，整个模型还堆叠多层，必须先说明统计范围。

**Q：现代 Decoder-only 模型每个 Block 也是三个残差连接吗？**

通常不是。常见的 Llama 风格 Decoder-only Block 主要围绕 Self-Attention 和 MLP 两个子层建立残差连接，没有原版 Cross-Attention 子层。

**③ 来源**

[牛客｜字节多模态搜索算法面经分享](https://www.nowcoder.com/feed/main/detail/67312a5472be4f85812eed8c9a67dffa) 中直接出现了 Transformer 残差连接数量及位置的问题。（原聊天内部引用，原链接待补核）

仓库对应：`day08/notes.md`、`day09/notes.md`

---

## Q024｜原版 Transformer 中，每一层 Decoder 都会读取 Encoder 输出吗？还是只有第一层读取？

(A · 重点) 架构细节补充题

**① 面试标准回答**

**原版 Transformer 中，每一层 Decoder 都包含 Cross-Attention，因此每一层都可以读取 Encoder 最后一层产生的隐藏状态。**

Encoder 首先将整个输入序列编码成一组隐藏表示。

随后，每个 Decoder Block 中的 Cross-Attention 都使用：

- Q：当前 Decoder Block 的隐藏状态。
- K、V：Encoder 最终输出经过当前 Cross-Attention 层的线性投影。

需要注意：虽然不同 Decoder 层读取的是同一组 Encoder 最终隐藏表示，但它们通常使用各自独立的 K、V 投影参数，因此产生的 K、V 不一定相同。

**② 高频追问**

**Q：Decoder 的第 2 层 Cross-Attention，读取 Encoder 第 2 层的输出吗？**

不是。标准原版结构中，各层 Decoder 的 Cross-Attention 都读取 Encoder Stack 的最终输出，而不是与 Encoder 层号一一对应。

**Q：训练时需要为 Decoder 的每一层重新运行 Encoder 吗？**

不需要。Encoder 的输出可以一次计算并供多个 Decoder Block 使用。

**③ 来源**

属于原版架构细节补充题；Harvard 的 Transformer 论文注释实现中，可以直接看到每个 Decoder Layer 接收同一 `memory` 的逻辑。（原聊天内部引用，原链接待补核）

仓库对应：`day09/code.py`

---

# 模块 01｜查漏与结项

Transformer 基础原理与整体架构

# 24 题

(基础模块完成)

Q001–Q024，已完成本模块预定知识范围的首轮整理；不代表所有公开面经已被穷尽检索。

## 一、24 道题的覆盖检查

| 知识领域 | 对应题号 | 结果 |
|--------|--------|--------|
| Transformer 设计动机、总体架构 | Q001–Q002、Q008 | 已覆盖 |
| Encoder / Decoder / Cross-Attention | Q003–Q005、Q024 | 已覆盖 |
| 并行训练、推理区别 | Q006、Q013 | 已覆盖 |
| 位置机制、Mask | Q007、Q014 | 基础已覆盖 |
| 时间复杂度、空间复杂度 | Q009 | 基础已覆盖 |
| Token → Embedding → Logits | Q010–Q012、Q015 | 已覆盖 |
| FFN、残差、Dropout、归一化 | Q016–Q017、Q019–Q021、Q023 | 基础已覆盖 |
| 多层堆叠、原版参数配置 | Q018、Q022 | 已覆盖 |

## 二、哪些问题没有在模块 01 展开？

不是遗漏，而是为了避免重复，明确留给后续模块：

| 后续模块 | 重点保留问题 |
|--------|--------|
| 02 · Attention 与 QKV | 为什么要 Q/K/V、缩放因子推导、Softmax、注意力矩阵 |
| 03 · 多头注意力 | Head 拆分、MHA/MQA/GQA、维度与信息表达 |
| 04 · 位置编码 | RoPE 数学推导、ALiBi、NTK Scaling、YaRN |
| 05 · FFN 与归一化 | LayerNorm/RMSNorm 公式、Pre-LN 梯度、SwiGLU |
| 06 · 模型架构比较 | BERT/GPT/T5 的训练目标与架构差异 |
| 07 · 现代大模型 | Llama、Qwen、MoE 等具体架构 |
| 08–09 · 推理优化 | KV Cache、FlashAttention、Prefill/Decode、显存 |
| 10–11 · 训练与代码 | Loss、参数量计算、手写 Transformer、Shape 推导 |

还有一个需要纠正的地方：此前 Q009 的复杂度解释中，不能笼统认为 Attention 始终是 Transformer 中计算量最大的部分。在不同序列长度和隐藏维度下，FFN、QKV 投影等部分也可能占有更高计算量。正式总文档会保留严谨版本。

