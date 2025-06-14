# LLM_InFeed
## Large Language Model Stance Detection with Hidden States Feedback  
## 基于大语言模型隐状态反馈的立场检测方法

### Abstract
Stance detection is a core task in natural language processing, aiming to identify the stance tendency expressed in the text towards a specific target. In low-resource scenarios, configuring a classification head for lightweight large language models to achieve efficient text classification holds broad application prospects. However, such methods still face two major challenges: first, lightweight large models have limited depth, making it difficult to support in-depth reasoning for complex stance texts; second, existing large language models are mostly pre-trained for generation tasks, and the deep Transformer layers have problems of information redundancy and low reasoning efficiency in classification tasks, which limits the effective utilization of their semantic capabilities. To address these issues, this paper proposes a multi-round feedback enhancement framework. By introducing a feedback transformer, the semantic information in the deep hidden states at the model's end is compressed and injected back into the intermediate layers, constructing multi-round reasoning paths and endowing the model with reasoning capabilities similar to human "repeated deliberation". This enhances the reasoning depth without significantly increasing model parameters and activates the inefficient parameters at the end. Experiments on multiple public stance detection datasets demonstrate that this method significantly improves classification performance while maintaining high efficiency, showing good practicality and application potential.  
立场检测是自然语言处理中的核心任务，旨在识别文本对特定目标所表达的立场倾向。在低资源场景下，为轻量级大语言模型配置分类头以实现高效文本分类具有广阔应用前景。然而，此类方法仍面临两大挑战：其一，轻量级大模型深度有限，难以支撑对复杂立场文本的深入推理；其二，现有大语言模型多以生成任务为预训练目标，深层 Transformer 层在分类任务中存在信息冗余与推理效率低下的问题，限制了其语义能力的有效发挥。为此，本文提出一种多轮反馈增强框架，通过引入反馈变换器，将模型末端深层隐状态中的语义信息压缩后注入回中间层，构建多轮推理路径，赋予模型类似人类“反复推敲”的推理能力，在不显著增加模型参数的前提下提升了推理深度，同时激活了末端低效参数。在多个公开立场检测数据集上的实验证明，该方法在保持高效率的同时显著提升了分类性能，展现出良好的实用性与应用潜力。  

![The Framwork of our Method](./Feedback.png "Framwork")

### Start
transformers==4.46.3

### Contect us  
Wang Yinglong wangyinglong2023@gmail.com
