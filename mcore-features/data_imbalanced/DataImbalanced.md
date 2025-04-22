# Data Imbalanced

<br><br>

## 1.数据不均衡的场景 

### 参考资料

1. [DistTrain: Addressing Model and Data Heterogeneity with Disaggregated Training for Multimodal Large Language Models](https://arxiv.org/abs/2408.04275)
2. [OmniBal: Towards Fast Instruct-tuning for Vision-Language Models via Omniverse Computation Balance](https://arxiv.org/abs/2407.20761)

### 数据不均衡
1. 同一mbs内部，不同的数据并行组，由于输入数据（images/videos/text）的不均衡导致计算的不均衡：Ref: [DistTrain](https://arxiv.org/abs/2408.04275)
![data_imbalanced_intra_mbs_straggler](./images/data_imbalanced/data_imbalanced_intra_mbs_straggler.png)
1. 不同的mbs之间，由于数据的不均衡，造成pipeline的bubble，从而降低训练效率；
![data_imbalanced_inter_mbs_straggler](./images/data_imbalanced/data_imbalanced_inter_mbs_straggler.png)

## 2.数据

### 具体数据分析

以常用的[LLaVA-Pretrain-Dataset](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)为例，分析一下`Images`数据的不均衡现象。`Images`会经过`ImageEncoder`编码得到`Image tokens`。如果在Encode过程中，保留图像的原始分辨率（不resize成统一的分辨率），那么不同分辨率的图像，经过`ImageEncoder`编码之后会得到不同数量的`Image tokens`，$N_{\text{image-tokens}}=\lceil{w//p}\rceil*\lceil{h//p}\rceil$。这里$w$,$h$分别是输入图像的宽高，$p$是patch size。先从`Image tokens`看一下不同分辨率图像带来的不均衡。

参考：[LLavaDatasetAnalysis.ipynb](./LLavaDatasetAnalysis.ipynb)

1. image tokens数量小于2500的数据分布如图：
   ![x](images/data_imbalanced/data_imbalanced_llava_datasets_stat_lessthan2500.png)
2. image tokens超过3000的case有180例，占总体数据量的不足0.05%
3. image tokens数量在500-1500之间的占比99.4%


<br><br>

## 数据不均衡的解决方案 (TODO:列举常见处理方案)


<br><br>

## 数据不均衡的解决方案——Sequence Packing

Sequence Packing的原理如下图所示：
![Sequence Packing for NeVA](./images/data_imbalanced/sequence_packing_principle.png)



### 随机数据做Sequence Packing

### Sequence Packing之前先做预处理

### 参数选取

<br>

## 试验

### 试验1：单卡H20

* H20 96G, single GPU
* image_tiles: 1-20, images tokens: 256-5120
  
|packing sequence| time per sample (ms)|buffer size|packing sequence length|sequence length|
|:--------------:|:-------------------:|:---------:|:---------------------:|:-------------:|
|disabled|||||
|enabled |||||

<br>

### 试验2：4卡H20, TP1PP1DP4
* H20 96G, 4GPUs
* image_tiles: 1-20, images tokens: 256-5120


|packing sequence| time per sample (ms)|buffer size|packing sequence length|sequence length|
|:--------------:|:-------------------:|:---------:|:---------------------:|:-------------:|
|disabled|||||
|enabled |||||






****

## 试验

## 参考资料
1. [Sequence Packing](https://docs.nvidia.com/nemo-framework/user-guide/24.12/nemotoolkit/features/optimizations/sequence_packing.html)
2. [Sequence Packing for NeVA](https://docs.nvidia.com/nemo-framework/user-guide/24.12/nemotoolkit/multimodal/mllm/sequence_packing.html)