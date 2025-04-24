<div align="center">

# A Survey of Efficient Reasoning for Large Reasoning Models: Language, Multimodality, and Beyond
[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2503.21614)  [![Github](https://img.shields.io/badge/Synced-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://mp.weixin.qq.com/s/JfXG93uOt_lTAs1zECsEsw)
[![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/suzhaochen0110/status/1905461785693749709?s=46)

[![Awesome](https://awesome.re/badge.svg)](https://github.com/XiaoYee/Awesome_Efficient_LRM_Reasoning) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/XiaoYee/Awesome_Efficient_LRM_Reasoning?color=green) 

</div>

---

## 🔔 News

- [2025-04] We include AgentPrune, where efficient reasoning is important for agent system. 
- [2025-04] We include benchmarks for Efficient Reasoning: MME-CoT, S1-Bench, DUMB500.
- [2025-04] We add Mamba Reasoning models (e.g M1) and Hybrid models (e.g Mamba-Transformer) in Efficient Reasoning during Pre-training. It is naturally efficient to infer.
- [2025-04] We add a new "Model Merge" category in Efficient Reasoning during Inference. It is feasible to be a promising direction.
- [2025-04] 📢 Our work is reported by both [Synced (机器之心)](https://mp.weixin.qq.com/s/JfXG93uOt_lTAs1zECsEsw) and [Zhuanzhi (专知)](https://mp.weixin.qq.com/s/VsRIB-yiKjoY2yp-kAmUNw).
- [2025-03] 📢 Our work is reported by both [Deep Learning and NLP (深度学习自然语言处理)](https://mp.weixin.qq.com/s/0dSMJjxrtRwAijVQXDBCvw) and [Machine Learning and NLP (机器学习算法与自然语言处理)](https://mp.weixin.qq.com/s/xygYjKHetkm7nFT92fmQWA).
- [2025-03] We released our survey "[A Survey of Efficient Reasoning for Large Reasoning Models: Language, Multimodality, and Beyond](https://arxiv.org/abs/2503.21614)". This is the **first survey** for efficient reasoning of **Large Reasoning Models**, covering **language**, **multimodality**, **agent**, and **applications**. We provide several promising future directions in our survey.   
- [2025-03] We created this repository to maintain a paper list on Awesome-Efficient-LRM-Reasoning.

---

![Author](figs/author.png)

![Taxonomy](figs/figure2.png)

> If you find our survey useful for your research, please consider citing:

```
@article{qu2025survey,
  title={A Survey of Efficient Reasoning for Large Reasoning Models: Language, Multimodality, and Beyond},
  author={Qu, Xiaoye and Li, Yafu and Su, Zhaochen and Sun, Weigao and Yan, Jianhao and Liu, Dongrui and Cui, Ganqu and Liu, Daizong and Liang, Shuxian and He, Junxian and others},
  journal={arXiv preprint arXiv:2503.21614},
  year={2025}
}

```


---

![Category](figs/category.png)


## 🔥 Table of Contents

- [Awesome-Efficient-LRM-Reasoning](#awesome-efficient-lrm-reasoning)
  - [👀 Introduction](#-introduction)
  - [💭 Efficient Reasoning during Inference](#-efficient-reasoning-during-inference)
    - [Length Budgeting](#length-budgeting)
    - [System Switch](#system-switch)
    - [Model Switch](#model-switch)
    - [Model Merge](#model-merge)
    - [Parallel Search](#parallel-search)
  - [💫 Efficient Reasoning with SFT](#-efficient-reasoning-with-sft)
    - [Reasoning Chain Compression](#reasoning-chain-compression)
    - [Latent-Space SFT](#latent-space-sft)
  - [🧩 Efficient Reasoning with Reinforcement Learning](#-efficient-reasoning-with-reinforcement-learning)
    - [Efficient Reinforcement Learning with Length Reward](#efficient-reinforcement-learning-with-length-reward)
    - [Efficient Reinforcement Learning without Length Reward](#efficient-reinforcement-learning-without-length-reward)
  - [💬 Efficient Reasoning during Pre-training](#-efficient-reasoning-during-pre-training)
    - [Pretraining with Latent Space](#pretraining-with-latent-space)
    - [Subquadratic Attention](#subquadratic-attention)
    - [Linearization](#linearization)
    - [Efficient Reasoning with Subquadratic Attention](#efficient-reasoning-with-subquadratic-attention)
  - [🔖 Future Directions](#-future-directions)
    - [Efficient Multimodal Reasoning and Video Reasoning](#efficient-multimodal-reasoning-and-video-reasoning)
    - [Efficient Test-time Scaling and Infinity Thinking](#efficient-test-time-scaling-and-infinity-thinking)
    - [Efficient and Trustworthy Reasoning](#efficient-and-trustworthy-reasoning)
    - [Building Efficient Reasoning Applications](#building-efficient-reasoning-applications)
    - [Evaluation and Benchmark](#evaluation-and-benchmark)
---

## 📜Content


### 👀 Introduction

In the age of LRMs, we propose that "**Efficiency is the essence of intelligence.**"
Just as a wise human knows when to stop thinking and start deciding, a wise model should know when to halt unnecessary deliberation. 
An intelligent model should manipulate the token economy, i.e., allocating tokens purposefully, skipping redundancy, and optimizing the path to a solution. Rather than naively traversing every possible reasoning path, it should emulate a master strategist, balancing cost and performance with elegant precision.

To summarize, this survey makes the following key contributions to the literature:
- Instead of offering a general overview of LRMs, we focus on the emerging and critical topic of **efficient reasoning** in LRMs, providing an in-depth and targeted analysis.
- We identify and characterize common patterns of reasoning inefficiency, and outline the current challenges that are unique to improving reasoning efficiency in large models.
- We provide a comprehensive review of recent advancements aimed at enhancing reasoning efficiency, structured across the end-to-end LRM development pipeline, from pretraining and supervised fine-tuning to reinforcement learning and inference.


---

## 🚀 Papers


### 💭 Efficient Reasoning during Inference

#### Length Budgeting

- [Thought Manipulation: External Thought Can Be Efficient for Large Reasoning Models](https://arxiv.org/abs/2504.13626) ![](https://img.shields.io/badge/abs-2025.04-red)
- [Reasoning Models Can Be Effective Without Thinking](https://arxiv.org/abs/2504.09858) ![](https://img.shields.io/badge/abs-2025.04-red)
- [How Well do LLMs Compress Their Own Chain-of-Thought? A Token Complexity Approach](https://arxiv.org/abs/2503.01141) ![](https://img.shields.io/badge/abs-2025.03-red)
- [Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching](https://arxiv.org/abs/2503.05179) ![](https://img.shields.io/badge/abs-2025.03-red)
- [Chain of Draft: Thinking Faster by Writing Less](https://arxiv.org/abs/2502.18600) ![](https://img.shields.io/badge/abs-2025.02-red)
- [SafeChain: Safety of Language Models with Long Chain-of-Thought Reasoning Capabilities](https://arxiv.org/abs/2502.12025) ![](https://img.shields.io/badge/abs-2025.02-red)
- [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393) ![](https://img.shields.io/badge/abs-2025.01-red)
- [Token-budget-aware llm reasoning](https://arxiv.org/abs/2412.18547) ![](https://img.shields.io/badge/abs-2024.12-red)
- [Efficiently Serving LLM Reasoning Programs with Certaindex](https://arxiv.org/abs/2412.20993) ![](https://img.shields.io/badge/abs-2024.12-red)
- [Make every penny count: Difficulty-adaptive self-consistency for cost-efficient reasoning](https://arxiv.org/abs/2408.13457) ![](https://img.shields.io/badge/abs-2024.08-red)
- [Scaling llm test-time compute optimally can be more effective than scaling model parameters](https://arxiv.org/abs/2408.03314) ![](https://img.shields.io/badge/abs-2024.08-red)
- [Concise thoughts: Impact of output length on llm reasoning and cost](https://arxiv.org/abs/2407.19825) ![](https://img.shields.io/badge/abs-2024.07-red)
- [The impact of reasoning step length on large language models](https://arxiv.org/abs/2401.04925v3) ![](https://img.shields.io/badge/abs-2024.01-red)
- [The benefits of a concise chain of thought on problem-solving in large language models](https://arxiv.org/abs/2401.05618) ![](https://img.shields.io/badge/abs-2024.01-red)
- [Guiding language model reasoning with planning tokens](https://arxiv.org/abs/2310.05707) ![](https://img.shields.io/badge/abs-2023.10-red)

#### System Switch

- [Fast-Slow-Thinking: Complex Task Solving with Large Language Models](https://arxiv.org/abs/2504.08690) ![](https://img.shields.io/badge/abs-2025.04-red)
- [Think More, Hallucinate Less: Mitigating Hallucinations via Dual Process of Fast and Slow Thinking](https://arxiv.org/abs/2501.01306) ![](https://img.shields.io/badge/abs-2025.01-red)
- [Dualformer: Controllable Fast and Slow Thinking by Learning with Randomized Reasoning Traces](https://arxiv.org/abs/2410.09918) ![](https://img.shields.io/badge/abs-2024.10-red)
- [Visual Agents as Fast and Slow Thinkers](https://arxiv.org/abs/2408.08862) ![](https://img.shields.io/badge/abs-2024.08-red)
- [System-1.x: Learning to Balance Fast and Slow Planning with Language Models](https://arxiv.org/abs/2407.14414) ![](https://img.shields.io/badge/abs-2024.07-red)
- [DynaThink: Fast or slow? A dynamic decision-making framework for large language models](https://arxiv.org/abs/2407.01009) ![](https://img.shields.io/badge/abs-2024.07-red)

#### Model Switch

- [SplitReason: Learning To Offload Reasoning](https://arxiv.org/abs/2504.16379) ![](https://img.shields.io/badge/abs-2025.04-red)
- [SpecReason: Fast and Accurate Inference-Time Compute via Speculative Reasoning](https://arxiv.org/abs/2504.07891) ![](https://img.shields.io/badge/abs-2025.04-red)
- [MixLLM: Dynamic Routing in Mixed Large Language Models](https://arxiv.org/abs/2502.18482) ![](https://img.shields.io/badge/abs-2025.02-red)
- [Closer Look at Efficient Inference Methods: A Survey of Speculative Decoding](https://arxiv.org/abs/2411.13157) ![](https://img.shields.io/badge/abs-2024.11-red)
- [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) ![](https://img.shields.io/badge/abs-2024.06-red)
- [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665) ![](https://img.shields.io/badge/abs-2024.06-red)
- [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710) ![](https://img.shields.io/badge/abs-2024.04-red)
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) ![](https://img.shields.io/badge/abs-2024.01-red)
- [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) ![](https://img.shields.io/badge/abs-2024.01-red)
- [Routing to the Expert: Efficient Reward-guided Ensemble of Large Language Models](https://arxiv.org/abs/2311.08692) ![](https://img.shields.io/badge/abs-2023.11-red)
- [Speculative Decoding with Big Little Decoder](https://arxiv.org/abs/2302.07863) ![](https://img.shields.io/badge/abs-2023.02-red)

#### Model Merge

- [Unlocking efficient long-to-short llm reasoning with model merging](https://arxiv.org/abs/2503.20641) ![](https://img.shields.io/badge/abs-2025.03-red)

#### Parallel Search

- [Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding](https://arxiv.org/abs/2503.01422) ![](https://img.shields.io/badge/abs-2025.03-red)
- [Efficient Test-Time Scaling via Self-Calibration](https://arxiv.org/abs/2503.00031) ![](https://img.shields.io/badge/abs-2025.03-red)
- [Scalable Best-of-N Selection for Large Language Models via Self-Certainty](https://arxiv.org/abs/2502.18581) ![](https://img.shields.io/badge/abs-2025.02-red)
- [Meta-Reasoner: Dynamic Guidance for Optimized Inference-time Reasoning in Large Language Models](https://arxiv.org/abs/2502.19918) ![](https://img.shields.io/badge/abs-2025.02-red)
- [Test-Time Preference Optimization: On-the-Fly Alignment via Iterative Textual Feedback](https://arxiv.org/abs/2501.12895) ![](https://img.shields.io/badge/abs-2025.01-red)
- [Fast Best-of-N Decoding via Speculative Rejection](https://arxiv.org/abs/2410.20290) ![](https://img.shields.io/badge/abs-2024.10-red)
- [TreeBoN: Enhancing Inference-Time Alignment with Speculative Tree-Search and Best-of-N Sampling](https://arxiv.org/abs/2410.16033) ![](https://img.shields.io/badge/abs-2024.10-red)
- [Scaling llm test-time compute optimally can be more effective than scaling model parameters](https://arxiv.org/abs/2408.03314) ![](https://img.shields.io/badge/abs-2024.08-red)

### 💫 Efficient Reasoning with SFT

#### Reasoning Chain Compression

- [Z1: Efficient Test-time Scaling with Code](https://arxiv.org/abs/2504.00810) ![](https://img.shields.io/badge/abs-2025.04-red)
- [Self-Training Elicits Concise Reasoning in Large Language Models](https://arxiv.org/abs/2502.20122) ![](https://img.shields.io/badge/abs-2025.02-red)
- [TokenSkip: Controllable Chain-of-Thought Compression in LLMs](https://arxiv.org/abs/2502.12067) ![](https://img.shields.io/badge/abs-2025.02-red)
- [Stepwise Perplexity-Guided Refinement for Efficient Chain-of-Thought Reasoning in Large Language Models](https://arxiv.org/abs/2502.13260) ![](https://img.shields.io/badge/abs-2025.02-red)
- [C3oT: Generating Shorter Chain-of-Thought without Compromising Effectiveness](https://arxiv.org/abs/2412.11664) ![](https://img.shields.io/badge/abs-2024.12-red)
- [Can Language Models Learn to Skip Steps?](https://arxiv.org/abs/2411.01855) ![](https://img.shields.io/badge/abs-2024.11-red)
- [Distilling System 2 into System 1](https://arxiv.org/abs/2407.06023) ![](https://img.shields.io/badge/abs-2024.07-red)


#### Latent-Space SFT

- [Beyond Chains of Thought: Benchmarking Latent-Space Reasoning Abilities in Large Language Models](https://arxiv.org/abs/2504.10615) ![](https://img.shields.io/badge/abs-2025.04-red)
- [From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step](https://arxiv.org/abs/2405.14838) ![](https://img.shields.io/badge/abs-2024.05-red)
- [CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation](https://arxiv.org/abs/2502.21074) ![](https://img.shields.io/badge/abs-2025.02-red)
- [Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning](https://arxiv.org/abs/2502.03275) ![](https://img.shields.io/badge/abs-2025.02-red)
- [SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs](https://arxiv.org/abs/2502.12134) ![](https://img.shields.io/badge/abs-2025.02-red)
- [LightThinker: Thinking Step-by-Step Compression](https://arxiv.org/abs/2502.15589) ![](https://img.shields.io/badge/abs-2025.02-red)
- [Efficient Reasoning with Hidden Thinking](https://arxiv.org/abs/2501.19201) ![](https://img.shields.io/badge/abs-2025.01-red)
- [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769) ![](https://img.shields.io/badge/abs-2024.12-red)
- [Compressed Chain of Thought: Efficient Reasoning Through Dense Representations](https://arxiv.org/abs/2412.13171) ![](https://img.shields.io/badge/abs-2024.12-red)

  
### 🧩 Efficient Reasoning with Reinforcement Learning

#### Efficient Reinforcement Learning with Length Reward

- [HAWKEYE: Efficient Reasoning with Model Collaboration](https://arxiv.org/pdf/2504.00424v1) ![](https://img.shields.io/badge/abs-2025.04-red)
- [ThinkPrune: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning](https://arxiv.org/abs/2504.01296) ![](https://img.shields.io/badge/abs-2025.04-red)
- [Think When You Need: Self-Adaptive Chain-of-Thought Learning](https://arxiv.org/abs/2504.03234) ![](https://img.shields.io/badge/abs-2025.04-red)
- [DAST: Difficulty-Adaptive Slow-Thinking for Large Reasoning Models](https://arxiv.org/abs/2503.04472) ![](https://img.shields.io/badge/abs-2025.03-red)
- [L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning](https://www.arxiv.org/abs/2503.04697) ![](https://img.shields.io/badge/abs-2025.03-red)
- [Demystifying Long Chain-of-Thought Reasoning in LLMs](https://arxiv.org/abs/2502.03373) ![](https://img.shields.io/badge/abs-2025.02-red)
- [Training Language Models to Reason Efficiently](https://arxiv.org/abs/2502.04463) ![](https://img.shields.io/badge/abs-2025.02-red)
- [O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning](https://arxiv.org/abs/2501.12570) ![](https://img.shields.io/badge/abs-2025.01-red)
- [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599) ![](https://img.shields.io/badge/abs-2025.01-red)
  
#### Efficient Reinforcement Learning without Length Reward

- [Concise Reasoning via Reinforcement Learning](https://arxiv.org/abs/2504.05185v1) ![](https://img.shields.io/badge/abs-2025.04-red)
- [Optimizing Test-Time Compute via Meta Reinforcement Fine-Tuning](https://arxiv.org/abs/2503.07572) ![](https://img.shields.io/badge/abs-2025.03-red)
- [Think Smarter not Harder: Adaptive Reasoning with Inference Aware Optimization](https://arxiv.org/abs/2501.17974) ![](https://img.shields.io/badge/abs-2025.01-red)
- [Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs](https://arxiv.org/abs/2412.21187) ![](https://img.shields.io/badge/abs-2024.12-red)

### 💬 Efficient Reasoning during Pre-training

#### Pretraining with Latent Space

- [LLM Pretraining with Continuous Concepts](https://arxiv.org/abs/2502.08524) ![](https://img.shields.io/badge/abs-2025.02-red)
- [Scalable Language Models with Posterior Inference of Latent Thought Vectors](https://arxiv.org/abs/2502.01567) ![](https://img.shields.io/badge/abs-2025.02-red)
- [Byte latent transformer: Patches scale better than tokens](https://arxiv.org/abs/2412.09871) ![](https://img.shields.io/badge/abs-2024.12-red)
- [Large Concept Models: Language Modeling in a Sentence Representation Space](https://arxiv.org/abs/2412.08821) ![](https://img.shields.io/badge/abs-2024.12-red)

#### Subquadratic Attention

- [RWKV-7 "Goose" with Expressive Dynamic State Evolution](https://arxiv.org/abs/2503.14456) ![](https://img.shields.io/badge/abs-2025.03-red)
- [LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid](https://arxiv.org/abs/2502.07563) ![](https://img.shields.io/badge/abs-2025.02-red)
- [Native sparse attention: Hardware-aligned and natively trainable sparse attention](https://arxiv.org/abs/2502.11089) ![](https://img.shields.io/badge/abs-2025.02-red)
- [MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189) ![](https://img.shields.io/badge/abs-2025.02-red)
- [MoM: Linear Sequence Modeling with Mixture-of-Memories](https://www.arxiv.org/abs/2502.13685) ![](https://img.shields.io/badge/abs-2025.02-red)
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) ![](https://img.shields.io/badge/abs-2024.12-red)
- [Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality](https://arxiv.org/abs/2405.21060) ![](https://img.shields.io/badge/abs-2024.05-red)
- [Various Lengths, Constant Speed: Efficient Language Modeling with Lightning Attention](https://arxiv.org/abs/2405.17381) ![](https://img.shields.io/badge/abs-2024.05-red)
- [Gated linear attention transformers with hardware-efficient training](https://arxiv.org/abs/2312.06635) ![](https://img.shields.io/badge/abs-2023.12-red)

#### Linearization

- [Liger: Linearizing Large Language Models to Gated Recurrent Structures](https://arxiv.org/abs/2503.01496) ![](https://img.shields.io/badge/abs-2025.03-red)
- [Llamba: Scaling Distilled Recurrent Models for Efficient Language Processing](https://arxiv.org/abs/2502.14458) ![](https://img.shields.io/badge/abs-2025.02-red)
- [LoLCATs: On Low-Rank Linearizing of Large Language Models](https://arxiv.org/abs/2410.10254) ![](https://img.shields.io/badge/abs-2024.10-red)
- [The Mamba in the Llama: Distilling and Accelerating Hybrid Models](https://arxiv.org/abs/2408.15237) ![](https://img.shields.io/badge/abs-2024.08-red)

#### Efficient Reasoning with Subquadratic Attention

- [M1: Towards Scalable Test-Time Compute with Mamba Reasoning Models](https://arxiv.org/abs/2504.10449) ![](https://img.shields.io/badge/abs-2025.04-red)
- [Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models](https://arxiv.org/abs/2504.03624v1) ![](https://img.shields.io/badge/abs-2025.04-red)
- [Compositional Reasoning with Transformers, RNNs, and Chain of Thought](https://arxiv.org/abs/2503.01544) ![](https://img.shields.io/badge/abs-2025.03-red)
- [Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning](https://arxiv.org/abs/2503.15558v1) ![](https://img.shields.io/badge/abs-2025.03-red)
- [Thinking Slow, Fast: Scaling Inference Compute with Distilled Reasoners](https://arxiv.org/abs/2502.20339) ![](https://img.shields.io/badge/abs-2025.02-red)


### 🔖 Future Directions

#### Efficient Multimodal Reasoning and Video Reasoning

- [Can Atomic Step Decomposition Enhance the Self-structured Reasoning of Multimodal Large Models?](https://arxiv.org/abs/2503.06252) ![](https://img.shields.io/badge/abs-2025.03-red)
- [Skywork R1V: Pioneering Multimodal Reasoning with Chain-of-Thought](https://huggingface.co/Skywork/Skywork-R1V-38B)
  
#### Efficient Test-time Scaling and Infinity Thinking

- [Efficient Test-Time Scaling via Self-Calibration](https://arxiv.org/abs/2503.00031) ![](https://img.shields.io/badge/abs-2025.03-red)
- [Dynamic self-consistency: Leveraging reasoning paths for efficient llm sampling](https://arxiv.org/abs/2408.17017) ![](https://img.shields.io/badge/abs-2024.08-red)

#### Efficient and Trustworthy Reasoning

- [X-Boundary: Establishing Exact Safety Boundary to Shield LLMs from Multi-Turn Jailbreaks without Compromising Usability](https://arxiv.org/abs/2502.09990) ![](https://img.shields.io/badge/abs-2025.02-red)
- [Deliberative alignment: Reasoning enables safer language models](https://arxiv.org/abs/2412.16339) ![](https://img.shields.io/badge/abs-2024.12-red)

#### Building Efficient Reasoning Applications (RAG, Tool, Agent)

- [The Danger of Overthinking: Examining the Reasoning-Action Dilemma in Agentic Tasks](https://arxiv.org/abs/2502.08235) ![](https://img.shields.io/badge/abs-2025.02-red)
- [Chain-of-Retrieval Augmented Generation](https://arxiv.org/abs/2501.14342) ![](https://img.shields.io/badge/abs-2025.01-red)
- [Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems](https://arxiv.org/abs/2410.02506) ![](https://img.shields.io/badge/abs-2024.10-red)

#### Evaluation and Benchmark

- [THOUGHTTERMINATOR: Benchmarking, Calibrating, and Mitigating Overthinking in Reasoning Models](https://arxiv.org/abs/2504.13367v1) ![](https://img.shields.io/badge/abs-2025.04-red)
- [S1-Bench: A Simple Benchmark for Evaluating System 1 Thinking Capability of Large Reasoning Models](https://arxiv.org/abs/2504.10368) ![](https://img.shields.io/badge/abs-2025.04-red)
- [DNA Bench: When Silence is Smarter -- Benchmarking Over-Reasoning in Reasoning LLMs](https://arxiv.org/abs/2503.15793) ![](https://img.shields.io/badge/abs-2025.03-red)
- [MME-CoT: Benchmarking Chain-of-Thought in Large Multimodal Models for Reasoning Quality, Robustness, and Efficiency](https://arxiv.org/abs/2502.09621) ![](https://img.shields.io/badge/abs-2025.02-red)
- [Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs](https://arxiv.org/abs/2412.21187) ![](https://img.shields.io/badge/abs-2024.12-red)


---



## Resources

**Reading lists related to Efficient Reasoning**

- [hemingkx/Awesome-Efficient-Reasoning](https://github.com/hemingkx/Awesome-Efficient-Reasoning)
- [Eclipsess/Awesome-Efficient-Reasoning-LLMs](https://github.com/Eclipsess/Awesome-Efficient-Reasoning-LLMs)
- [Hongcheng-Gao/Awesome-Long2short-on-LRMs](https://github.com/Hongcheng-Gao/Awesome-Long2short-on-LRMs)
- [DevoAllen/Awesome-Reasoning-Economy-Papers](https://github.com/DevoAllen/Awesome-Reasoning-Economy-Papers)
- [Blueyee/Efficient-CoT-LRMs](https://github.com/Blueyee/Efficient-CoT-LRMs)
- [EIT-NLP/Awesome-Latent-CoT](https://github.com/EIT-NLP/Awesome-Latent-CoT)
- [yzhangchuck/awesome-llm-reasoning-long2short-papers](https://github.com/yzhangchuck/awesome-llm-reasoning-long2short-papers)


## 🎉 Contribution

### Contributing to this paper list

⭐" **Join us in improving this repository!** If you know of any important works we've missed, please contribute. Your efforts are highly valued!   "

### Contributors

<a href="https://github.com/XiaoYee/Awesome_Efficient_LRM_Reasoning/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=XiaoYee/Awesome_Efficient_LRM_Reasoning" />
</a>

---

<!-- ## ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=XiaoYee/Awesome_Efficient_LRM_Reasoning&type=Date)](https://star-history.com/#XiaoYee/Awesome_Efficient_LRM_Reasoning&Date) -->
