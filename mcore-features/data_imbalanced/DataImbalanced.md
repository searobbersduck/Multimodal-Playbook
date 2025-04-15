# Data Imbalanced

## 数据不均衡的场景 

### 参考资料

1. [DistTrain: Addressing Model and Data Heterogeneity with Disaggregated Training for Multimodal Large Language Models](https://arxiv.org/abs/2408.04275)
2. [OmniBal: Towards Fast Instruct-tuning for Vision-Language Models via Omniverse Computation Balance](https://arxiv.org/abs/2407.20761)

### 数据不均衡
1. 同一mbs内部，不同的数据并行组，由于输入数据（images/videos/text）的不均衡导致计算的不均衡：Ref: [DistTrain](https://arxiv.org/abs/2408.04275)
![data_imbalanced_intra_mbs_straggler](./images/data_imbalanced/data_imbalanced_intra_mbs_straggler.png)
1. 不同的mbs之间，由于数据的不均衡，造成pipeline的bubble，从而降低训练效率；
![data_imbalanced_inter_mbs_straggler](./images/data_imbalanced/data_imbalanced_inter_mbs_straggler.png)

### 具体数据分析

以常用的[LLaVA-Pretrain-Dataset](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)为例，分析一下数据的不均衡现象。
