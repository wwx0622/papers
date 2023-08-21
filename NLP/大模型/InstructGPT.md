# title
[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

[博客1: InstructGPT简介](https://zhuanlan.zhihu.com/p/626665665)

1. 通过人工手机的高质量<Prompt, Response>数据集用来训练一个有监督的微调模型，SFT
2. **半自动**的收集偏好数据集，训练Reward Model，用于后面RL精调时给Response打分。半自动是SFT对Prompt生成多个Response，由人工标注哪个好（并不是评分，只是评价哪个回答更好，哪个更差），然后由Reward Model打分，自然更好的回答分数更高。分值差表示偏向于好回答多余差的回答的概率。
3. 以Reward Model作为奖励函数，以SFT为初始策略，跑PPO改善策略表现，得到强化版本的模型PPO-ptx，这个版本也就是所谓的InstructGPT。
