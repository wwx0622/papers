# title

[Training a Helpful and Harmless Assistant with
Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)

# Abstract

我们应用偏好建模和来自人类反馈的强化学习（RLHF）来微调语言模型，以充当有用和无害的助手。我们发现，这种对齐训练可以提高几乎所有NLP评估的性能，并且与python编码和摘要等专业技能的训练完全兼容。我们探索了一种迭代的在线培训模式，其中偏好模型和RL策略每周都会用新的人类反馈数据进行更新，从而有效地改进我们的数据集和模型。最后，我们研究了RLHF训练的稳健性，并确定了RL奖励和策略及其初始化之间KL偏差的平方根之间的大致线性关系。

# 1 Introduction

在本文中，我们表明，通过收集人类偏好数据并应用偏好建模（PMing）和人类反馈强化学习（RLHF）技术，我们可以训练一个相对有用且无害的（HH）自然语言助手。我们的完整培训过程如下图所示。
![Alt text](../../image/大模型/HH%20with%20RLHF/RLHF%2001.png)

我们将有益性和无害性分开对待，分别收集不同的人类偏好数据集。为了提供帮助，我们要求众包工作者请求我们的模型协助任何纯粹基于文本的任务。为了无害，我们邀请众包工作者对抗性地探测或“红队”我们的语言模型，以引发有害的反应:要么帮助他们实现有害的目标，要么导致人工智能使用有害的语言在与人工智能助手对话的每个阶段，众包工作者都会得到两种可能的回答。那些参与助人任务的人被要求选择更有帮助和诚实(即更好)的回答。那些参与红色团队任务的人被指示要选择更有害(即更糟糕)的反应。这些对话和人类表达的偏好构成了我们的数据集。

有用和无害往往对立。过分无害会导致无法真正解决需求。同样的，过分有用可能产生有害反应。
在混合训练下，PMs可以学到正确的经验，在适当时候表现有用性与吴海星。然后使用PM得分训练助手，并且有用性的RHLF更容易被红队选中。

关于一致性训练经常被提出的一个问题是，它是否会损害人工智能的能力。我们发现，当RLHF应用于大型语言模型时，答案似乎几乎是否定的。我们的rlhf训练的模型在几乎所有的评估中都比原始的、生成的模型表现得更好。我们还认为，可以在不影响一致性或性能的情况下，将专业技能与一致性相关的培训混合在一起。在实践中，对齐的模型可能比原始的模型更易于用户使用和可部署，这表明没有理由部署没有为对齐进行微调的模型。

## 1.1 Contributions

***Dialogue Preference Datasets***

- 在我们的界面中，我们主要使用各种52B语言模型收集单独的有益和无害(即红队)数据集。众包工作者与模型进行开放式对话，要么寻求帮助，要么提供说明，要么试图让模型发出有害的反应，他们被要求在每个对话步骤中分别选择更有益的反应或更有害的反应。
- 我们收集了三组数据，分别来自初始模型、对早期偏好模型的拒绝抽样、通过人类反馈进行“在线”强化学习训练的模型，我们大约每周改进一次。

***Alignment with Human Values Has Many Benefits and Essentially No Cost to Performance***

- 小模型有较大的“对齐惩罚”，在RLHF后效果下降。但是“对齐奖励”同样丰厚，大模型的零样本、少样本能力提高。
- HH的RLHF可以用于提升编程能力。并且结合HH偏好和专业技能不会导致性能下降。
- 有用性和无害性对立。然而随着模型的增大，PMs可以同时在两个分部上有较好表现。
- 可以使用OOD检测技术来拒绝大多数奇怪和有害的请求

***Scaling, RLHF Robustness, and Iterated ‘Online’ Training***

- 我们研究了PM准确性的缩放关系，将其作为模型和数据集大小的函数，并发现了大致的对数线性趋势。
- 我们对RLHF的稳健性进行了实验：较大的PM比较小的PM更稳健，正如预期的那样，RLHF训练过程中过拟合增加
- 我们发现对于大部分RLHF训练，$\sqrt{D_{KL}(\pi||\pi_0)}$和奖励近似线性相关，其中π和π0分别是策略和初始策略。
- 我们研究迭代在线培训，每周更新我们的偏好模型和RLHF政策，然后重新部署这些新的RLHF模型与众包工作者互动。这显著改进了模型。

## 1.2 Summary of Evaluations and Metrics

- ***NLP和代码评估***：在除TriviaQA之外的所有情况下，12B和52B RLHF训练的模型都比基本LM表现更好。我们还将HH的PM训练与总结作为一项专业技能进行了混合实验，并评估了由此产生的PM性，发现混合训练不会降低PM准确性。
- ***Static Alignment Evaluations***：RLHF改善了对所有群体的情绪，但并不能消除偏见
- ***Human Evaluations***：我们根据众包工作者的偏好计算Elo分数。我们还测试了在线模型在训练期间的性能，比较了不同级别的拒绝采样，并对迭代在线训练进行了对照实验。此外，我们聘请了专业作家来撰写对话，由助理提供高质量、有用和诚实的回应，然后我们请众包工作者将我们的模型的回应与这些作家的回应进行比较。
- ***Samples***：我们提供了所有PALM和De敏感问题的样本，以及附录C中InstructGPT和LaMDA提供的提示。我们在第6.1节中展示了与人类作家的一些比较
