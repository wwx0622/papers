# Scaling Instruction-Finetuned Language Models

[Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)

## Abstract

在一组被表述为指令的数据集上微调语言模型已被证明可以提高模型性能和对未见过的任务的泛化能力。在本文中，我们探索了指令微调，特别关注

1. 缩放任务数量，
2. 缩放模型大小
3. 微调思想链数据。

<!--
TODO:
前置知识:
    了解下面几个大模型:
        1. PaLM
        2. T5
        3. U-PaLM
    了解下面几个提示设置:
        1. 零样本: 对训练集未出现的任务提供正确的答案
        2. 少样本: 不知道任务的类别，但是可以通过提供少量的输入输出示例对输入做出反应
        3. CoT
    了解下面几个评估指标:
        1. MMLU
        2. BBH
        3. TyDiQA
        4. MGSM
        5. 开放式生成
        6.  RealToxicityPrompts
-->

我们发现，具有上述方面的指令微调显著提高了各种模型类（PaLM、T5、U-PaLM）、提示设置（零样本、少速、CoT）和评估基准（MMLU、BBH、TyDiQA、MGSM、开放式生成、RealToxicityPromts）的性能。
例如，在1.8K任务上微调的Flan PaLM 540B指令的性能大大优于PaLM 540B（平均+9.4%）。Flan PaLM 540B在几个基准测试中实现了最先进的性能，例如在五次射击MMLU中的75.2%。我们还公开发布了Flan-T5检查点，1即使与更大的型号（如PaLM 62）相比，也能实现强大的少镜头性能

## Introduction

本文从以下几方面进行指令微调(instruction finetuning)：

1. 我们研究了尺度(Scaling)对指令微调的影响。我们的实验表明，指令微调可以很好地适应任务数量和模型大小。他们各自的尺度行为表明，未来的研究应该进一步扩大任务数量和模型的规模。
2. 我们研究了微调(finetuning)对模型执行推理任务能力的影响。我们的实验表明，尽管不包括思想链(CoT)的先前指令微调方法严重降低了CoT评估的性能，但在微调混合物中仅添加9个CoT数据集可以在所有评估中获得更好的性能。

本文使用540B的参数训练Flan-PaLM，微调任务数增至1.8K，并包含CoT数据。可以在若干测试中达到最优水平准。此外，在Flan-T5(80M 到 11B参数)模型的上做指令微调后获得了强大的零样本(Zero-Shot)、少样本(few-shot)和CoT能力。

## 2 Flan Finetuning

利用各种指令模版(见下图)对数据源进行指令微调。

![指令微调模版](../../image/Instruction-Finetuning%2001.png)

### 2.1 Finetuning Data

- ***Text mixtures***: 增加指令微调中的任务数量可以提高对未见任务的泛化。本文结合先前工作中的四种任务(Muffin、T0-SF、NIV2和CoT)，将调优任务扩展到1836个，如下图所示。

![任务](../../image/Instruction-Finetuning%2002.png)

- ***Chain-of-thought finetuning mixture***: 第四种调优数据混合(推理)涉及CoT注释，我们使用它来探索对CoT注释进行调优是否会提高未见过的推理任务的性能。我们从先前的工作中创建了九个数据集的新混合物，人类评分员为训练语料库手动编写了CoT注释。我们手动为每个任务编写10个指令模板。

- ***Templates and formatting***: 对于Muffin, T0-SF和NIV2，我们使用混合物创建者给出的每个任务的教学模板。对于CoT，我们为9个数据集中的每一个手动编写大约10个指令模板。为了创建少量模板，我们编写了各种范例分隔符(例如，“Q:”/“a:”)，并在示例级别随机应用它们。图3显示了一个使用和不使用示例以及使用和不使用CoT进行格式化的示例。

### Finetuning procedure

本文在广泛的模型族中应用指令微调，包括T5，PaLM和U-PaLM。这些型号家族跨越一系列尺度，从Flan-T5-small (80M参数)到PaLM和U-PaLM (540B参数)。对于每个模型，我们应用相同的训练过程，除了几个超参数:学习率、批大小、dropout和微调步骤。我们使用一个恒定的学习率计划，并使用Adafactor优化器进行微调。我们使用打包(packing)将多个训练示例组合成单个序列，使用序列结束令牌将输入与目标分开。屏蔽用于防止令牌在打包示例边界上关注其他令牌。对于每个模型，我们使用单个检查点进行所有评估;最优步骤是基于搁置任务的周期性评估(每2k到10k步，取决于模型大小)来选择的，我们在给定模型的所有消融运行中使用相同数量的检查点步骤。值得注意的是，与训练计算相比，用于调优的计算量只占很小的一部分。

### Evaluation protocol

***Evaluation benchmarkds***: 我们关注的是未包含在调优数据中的搁置任务的性能。我们对Flan-PaLM在世界知识和推理任务上的整体能力感兴趣。因此，我们在一系列不同的基准上评估模型，包括多语言基准。(1) MMLU包括数学、历史、法律和医学等57个任务的试题。(2) BBH包括来自BIG-Bench的23个具有挑战性的任务，PaLM在这些任务中的表现低于人类平均水平。(3) TyDiQA是一个跨8种不同类型语言的问答基准。(4) MGSM是手工翻译成10种语言的数学单词问题的多语言基准。

***Evaluation methods and metrics***: 

- 对于MMLU和BBH，我们评估了通过直接提示直接预测答案的能力，其中模型直接给出答案，以及通过思维链(CoT)提示，其中模型必须在给出最终答案之前提供推理链。
- 对于TyDiQA，我们只测量直接提示的精确匹配分数，因为突出显示具有正确答案的文章部分可能不需要复杂的推理。
- 对于MGSM，我们只测量CoT提示精度，因为直接提示的性能很低。对于所有基准测试，我们使用给定的少量示例，示例的数量遵循先前的工作:MMLU为5次，BBH为3次，TyDiQA为1次，MGSM为8次。

## Scaling to 540B parameters and 1.8K tasks

我们首先根据**模型的大小**和**调优任务的数量**来检查缩放对搁置任务性能的影响。
模型大小: 8B $\rightarrow$ 62B $\rightarrow$ 540B
混合任务: None $\rightarrow$ +CoT $\rightarrow$ +Muffin $\rightarrow$ +T0-SF $\rightarrow$ +NIV2

![实验结果1](../../image/Instruction-Finetuning%2003.png)

上图表明:

- 多任务指令微调大大提升了性能
- 增加微调任务可以提高性能
- 模型增加一个数量级后可以大大提升微调和非微调模型的性能

## Finetuning with chain-of-thought annotations

Flan调优的目标是在一系列评估中产生一个改进的检查点，除了传统的NLP任务外，还包括多步推理能力。在本节中，我们将探讨在指令调优组合中包含思想链(CoT)数据的效果。

### Finetuning on chain-of-thought improves reasoning on held-out tasks

我们首先表明，在微调混合中包含九个具有思维链(CoT)注释的数据集可以提高推理能力。表4显示了Flan-PaLM的CoT提示能力在四个hold out评估基准上优于PaLM。

![CoT对性能的提升](../../image/Instruction-Finetuning%2004.png)

- 上表还显示了CoT提示如何与自一致性在几个基准上实现新的最先进的性能。
- 与某些专用模型相比，Flan-PaLM无法实现SOTA。

### Some chain-of-thought data is needed to maintain reasoning ability

在指令微调中仅包含九个CoT数据集的效果。
数据集基准:

- held-out CoT基准（MMLU、BBH和MGSM）
- held-out non-CoT基准（MMLU、BBH和TyDiQA）

计算CoT和non-CoT的归一化平均值。

![CoT基准测试](../../image/Instruction-Finetuning%2005.png)

- 在左图，结合not-CoT和CoT微调的性能比单独的CoT微调更强。
- 在右图，与仅在non-CoT上进行微调相比，在组合CoT和non-CoT的微调不会影响非CoT任务的性能。
- 左图还显示，为了保持这种推理能力，对一些CoT示例进行微调至关重要，因为仅对not-CoT进行微调会使CoT的性能显著降低，如绿线所示。鉴于先前的多项研究发现，指令微调可以提高unseen任务的性能，这种退化可能令人惊讶。可以将这种消融解释为：当unseen任务与微调任务（即non-CoT或CoT）处于相同的提示范式时，指令微调可以改善unseen任务。因此，提升模型推理能力是需要non-CoT和CoT数据

### Unlocking zero-shot reasoning

不论有没有示例，对CoT数据进行指令微调的最终好处是，所得到的模型能够在zero-shot中执行CoT推理。这种零样本设置很重要，因为它测试了模型在没有CoT的少样本的情况下产生自己的推理技能的能力，这可能需要大量的即时工程来正确组合。

Flan-PaLM可以通过利用短语"let's think step-by-step"激活的CoT推理来提高性能。

尽管PaLM的零样本CoT阴性结果可能与Kojima等人的研究结果相矛盾。
，更仔细的比较表明，它们并不矛盾。该论文中大多数成功的零样本CoT实验事实上都利用了InstructionGPT，这是一种指令微调（我们假设这种指令微调包括一些CoT样数据）。

## Putting it all together

鉴于先前关于缩放任务数量和包括思想链数据的结果，我们现在通过将指令微调应用于不同规模、架构和训练目标的几个模型来展示其通用性。

指令微调大大提高了所有模型类型的标准化平均性能。对于没有指令微调的T5模型，我们使用LM自适应模型。考虑到我们的评估基准的困难以及T5不是多语言的事实，与非微调模型相比，T5模型从教学微调中受益最大。另一个亮点是，我们在本文中实现的最强整体模型将指令微调与U-PaLM模型中使用的UL2持续预训练相结合。这一结果表明，指令微调和UL2持续预训练是互补的计算效率

## Usability evaluation of open-ended generation

除了NLP基准之外，语言模型还能够生成对开放请求的长格式答案。标准NLP基准和用于评估它们的自动度量不足以衡量这些开放式反应中的人类偏好。因此，我们进行了一项手动评估，调查教学微调对模型对具有挑战性的输入做出开放式回应的能力的影响。为此，我们创建了一个包含190个示例的评估集。该评估集包括以零样本方式向模型提出的五个具有挑战性的问题类别，每组20个问题：创造力、上下文推理、复杂推理、计划和解释。对于其中的60个例子（来自复杂的推理、规划和解释类别），我们创建了一个具有思维链触发短语的变体（例如，“让我们循序渐进地思考”），作为对CoT上的微调是否启用零样本的另一个评估

这一人类评估的结果显示，79%的时间首选Flan-PaLM。对于每个零样本设置，FlanPaLM在很大程度上是首选的，而对于使用CoT触发短语的输入，评分者对Flan-PaLM的偏好比PaLM进一步增加了约10%。至于少样本，与PaLM相比没有回归。

通过一组标注器演示对语言模型进行微调，以及从人类反馈中进行强化学习，改善了人类对用户提示分布的评估。
此外，对PaLM的模型生成的检查揭示了尽管在NLP基准上有很强的性能，但仅在下一个标记预测目标上进行预训练对于良好的零样本可用性是不够的。例如，PaLM对不期望的行为进行的定性检查包括（1）继续生成相关文本而不是回答问题，（2）在进行小修改的情况下重复输入问题，以及（3）不知道何时停止生成文本。这可能是在预训练中不使用序列结束标记的产物。

## Discussion

***scaling curves for instruction finetuning***:指令微调的两个关键组成部分——模型的大小和微调任务的数量——可以提高性能。我们绘制了这两个组件的缩放曲线，这表明缩放模型的大小和微调任务的数量有望继续提高性能，尽管缩放任务的数量具有递减的（尽管仍然是正的）回报。此外，与没有微调的模型相比，指令微调的改进幅度似乎没有降低，这表明指令微调可能对未来的模型仍然有意义。

***CoT finetuning is critical for reasoning abilities***:尽管先前的指令微调工作表明，对非CoT任务的微调可以提高看不见的非CoT工作的性能，但我们发现这实际上会导致CoT任务性能下降。作为这种CoT性能下降的解决方案，我们对非CoT和CoT数据进行了联合微调（第4节）。这种联合微调实现了显著更好的CoT性能，同时保持了非CoT任务的性能，允许单个模型在所有评估中都表现良好。

***Instruction finetuning generalizes across models***: 指令微调具有一般性，即使架构、大小发生改变。

***Instruction finetuning improves usability and mitigates some potential harms***： 微调模型产生的输出更符合人类偏好。

***Instruction finetuning is relatively compute-efficient***: 尽管缩放语言模型的大小已被证明可以可靠地提高性能，但它需要大量的计算。因此，开发计算高效的技术是很重要的。

总之，上面的证据强调了指令微调如何在一系列少样本、零样本、CoT和开放式生成评估中提高性能。指令微调跨模型进行了概括，并与UL2R等其他技术很好地结合在一起。与模型预训练相比，所有这些好处都带来了相对较小的计算成本。
