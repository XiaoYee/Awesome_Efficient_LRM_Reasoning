# Awesome-Efficient-LRM-Reasoning

[![Awesome](https://awesome.re/badge.svg)](https://github.com/XiaoYee/Awesome_Efficient_LRM_Reasoning) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/XiaoYee/Awesome_Efficient_LRM_Reasoning?color=green) 

Must Read papers about Awesome-Efficient-LRM-Reasoning

---

## 🔔 News

[2025-03] We create this repository to maintain a paper list on Awesome-Efficient-LRM-Reasoning.

---

## 🔥 Table of Contents

- [Awesome-Efficient-LRM-Reasoning](#awesome-efficient-lrm-reasoning)
  - [🔔 News](#-news)
  - [🔥 Table of Contents](#-table-of-contents)
  - [📜Content](#content)
    - [Abstract](#abstract)
    - [Introduction](#introduction)
  - [🌄 Papers](#-papers)
    - [🤖 Reasoning Inefficiency: Definition, Patterns and Challenges](#-reasoning-inefficiency-definition-patterns-and-challenges)
      - [Patterns of Reasoning Inefficiency](#patterns-of-reasoning-inefficiency)
      - [Unique Challenges for Efficient Reasoning in the Era of LRMs](#unique-challenges-for-efficient-reasoning-in-the-era-of-lrms)
    - [💭 Efficient Reasoning during Inference](#-efficient-reasoning-during-inference)
      - [Length Budgeting](#length-budgeting)
      - [System Switch](#system-switch)
      - [Model Switch](#model-switch)
      - [Parallel Search](#parallel-search)
    - [💫 Efficient Reasoning with SFT](#-efficient-reasoning-with-sft)
      - [Reasoning Chain Compression](#reasoning-chain-compression)
      - [Latent-Space SFT](#latent-space-sft)
    - [🧩 Efficient Reasoning with Reinforcement Learning](#-efficient-reasoning-with-reinforcement-learning)
      - [Efficient Reinforcement Learning with Length Reward](#efficient-reinforcement-learning-with-length-reward)
      - [Efficient Reinforcement Learning without Length Reward](#efficient-reinforcement-learning-without-length-reward)
    - [💬 Efficient Reinforcement Learning without Length Reward](#-efficient-reinforcement-learning-without-length-reward)
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
  - [🎉 Contribution](#-contribution)
    - [Contributing to this paper list](#contributing-to-this-paper-list)
    - [Contributors](#contributors)
---

## 📜Content

### Abstract


**Recent Large Reasoning Models (LRMs)**, such as DeepSeek-R1 and OpenAI o1, have demonstrated strong performance gains by scaling up the length of Chain-of-Thought (CoT) reasoning during inference. However, a growing concern lies in their tendency to produce excessively long and inefficient reasoning traces, which are often filled with redundant content (e.g., repeated definitions), over-analysis of simple problems, and superficial exploration of multiple reasoning paths for harder tasks. 
This inefficiency introduces significant challenges for training, inference, and real-world deployment (e.g., in agent-based systems), where token economy is critical. 
In this survey, we provide a comprehensive overview of recent efforts aimed at improving reasoning efficiency in LRMs, with a particular focus on the unique challenges that arise in this new paradigm.
We identify common patterns of inefficiency, examine methods proposed across the LRM lifecycle, i.e., from pretraining to inference, and discuss promising future directions for research. 
To support ongoing development, we also maintain a real-time [GitHub repository](https://github.com/XiaoYee/Awesome_Efficient_LRM_Reasoning) tracking recent progress in the field.
We hope this survey serves as a foundation for further exploration and inspires innovation in this rapidly evolving area.

### Introduction

In the age of LRMs, we propose that "**Efficiency is the essence of intelligence.**"
Just as a wise human knows when to stop thinking and start deciding, a wise model should know when to halt unnecessary deliberation. 
An intelligent model should manipulate the token economy, i.e., allocating tokens purposefully, skipping redundancy, and optimizing the path to a solution. Rather than naively traversing every possible reasoning path, it should emulate a master strategist, balancing cost and performance with elegant precision.

To summarize, this survey makes the following key contributions to the literature:
- Instead of offering a general overview of LRMs, we focus on the emerging and critical topic of **efficient reasoning** in LRMs, providing an in-depth and targeted analysis.
- We identify and characterize common patterns of reasoning inefficiency, and outline the current challenges that are unique to improving reasoning efficiency in large models.
- We provide a comprehensive review of recent advancements aimed at enhancing reasoning efficiency, structured across the end-to-end LRM development pipeline, from pretraining and supervised fine-tuning to reinforcement learning and inference.


---

## 🌄 Papers

### 🤖 Reasoning Inefficiency: Definition, Patterns and Challenges

#### Patterns of Reasoning Inefficiency

#### Unique Challenges for Efficient Reasoning in the Era of LRMs

### 💭 Efficient Reasoning during Inference

#### Length Budgeting

#### System Switch

#### Model Switch

#### Parallel Search


### 💫 Efficient Reasoning with SFT

#### Reasoning Chain Compression

#### Latent-Space SFT

### 🧩 Efficient Reasoning with Reinforcement Learning

#### Efficient Reinforcement Learning with Length Reward

#### Efficient Reinforcement Learning without Length Reward

### 💬 Efficient Reinforcement Learning without Length Reward

#### Pretraining with Latent Space

#### Subquadratic Attention

#### Linearization

#### Efficient Reasoning with Subquadratic Attention

### 🔖 Future Directions

#### Efficient Multimodal Reasoning and Video Reasoning

#### Efficient Test-time Scaling and Infinity Thinking

#### Efficient and Trustworthy Reasoning

#### Building Efficient Reasoning Applications

#### Evaluation and Benchmark




---
## 🎉 Contribution

### Contributing to this paper list

⭐" **Join us in improving this repository!** If you know of any important works we've missed, please contribute. Your efforts are highly valued!   "

### Contributors

<a href="https://github.com/XiaoYee/Awesome_Efficient_LRM_Reasoning/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=XiaoYee/Awesome_Efficient_LRM_Reasoning" />
</a>
