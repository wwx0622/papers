# title

[Siren’s Song in the AI Ocean: A Survey on Hallucination in Large Language Models](https://arxiv.org/abs/2309.01219)

# Abstract

幻觉现象：LLMs偶尔会生成不符合输入、前后矛盾、不符合现实的回复。为LLM在真实场景中的可依赖性增加了挑战。

# Introduction

解决幻觉现象有以下问题：

- 需要大量数据
- LLMs的通用性
- 错误的不易察觉

# Hallucination in the Era of LLM

幻觉类型：

- Input-conflicting
- Context-conflicting
- Fact-conflicting

LLM的其它问题:
- Ambiguity
- Incompleteness
- Bias
- Under-informativeness

# Evaluation of LLM Hallucination

## Evaluation Benchmarks

- Evaluation format:
  - generate（生成）
  - discriminate（选择）
- Task format
- Construction methods

## Evaluation Metrics

- Human Evaluation
- Model-based automatic evaluation
- Rule-based automatic evaluation：例如准确率、Rouge-L等

# Source of LLM Hallucination

- LLMs lack relevant knowledge or internalize false knowledge
- LLMs sometimes overestimate their capabilities
- Problematic alignment process could mislead LLMs into hallucination
- The generation strategy employed by LLMs has potential risks

# mitigation of LLM Hallucination


- Mitigation during Pre-train:聚焦预训练数据
- Mitigation during SFT:聚焦于监督微调数据
- Mitigation during RLHF:1)训练Reward model;2)训练SFT通过PPO
- Mitigation during Inference:
  - Designing Decoding Strategies：确定如何从概率分布中输出token
  - Resorting to External Knowledge: 使用外部知识作为补充
    - Knowledge acquistion：
      - External knowledge base：如数据库、语料库、维基百科等
      - External knowledge tools：如搜索引擎、代码执行器、谷歌学术接口等
    - Knowledge utilization：Generation-time supplement和Post-hoc correction
    - 利用外部指示带来的挑战：
      - knowledge verification
      - performance/efficiency of retriever/fixer
      - knowledge conflict
  - Exploiting Uncertainty
    - Logit-based estimation
    - verbalize-based estimation
    - consistency-based estimation
- other methods:
  - multi-agent interaction
  - prompt engineering
  - analyzing LLMs' internal states
  - Human-in-the-loop
  - Optimizing model architecture

总结：

![Alt text](image-1.png)
