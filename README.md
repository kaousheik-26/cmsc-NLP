# CMSC 723: LLM Safety & Jailbreak Analysis

This repository contains the code, data, and analysis for the **CMSC 723: Natural Language Processing** course project. The focus of this project is to evaluate and analyze the safety alignment of **Qwen3** language models (0.6B and 4B parameters) against various adversarial attacks and benchmarks.

## 📌 Project Overview

Large Language Models (LLMs) are increasingly deployed in real-world applications, making safety alignment critical. This project investigates:
* **Refusal Rates:** How often models refuse malicious prompts across different benchmarks.
* **Model Scaling:** Comparing safety performance between small (0.6B) and medium (4B) models.
* **Attack Vectors:** Analyzing susceptibility to jailbreaks like **AutoDAN** (Hierarchical Genetic Algorithms) and standard refusal string matching.

## 📊 Benchmarks Used

We evaluate models on the following datasets:
* **JailbreakBench**
* **InTheWild**
* **AdvBench**
* **StrongREJECT**
* **HarmBench**

## 🏗️ Models Evaluated

The following configurations of Qwen3 were tested:
* **Base Models:** Qwen3-0.6B, Qwen3-4B
* **Judge Variants:**
    * `openai/gpt-oss-20b`
    * `Qwen/Qwen3-14B`
    * `Qwen/Qwen3-30B-A3B`

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/kaousheik-26/cmsc-NLP.git](https://github.com/kaousheik-26/cmsc-NLP.git)
    cd cmsc-NLP
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch trl transformers pandas 
    ```

## 💻 Usage

### Running Evaluations
To run training edit the model and dataset in the train_trl.py and start training with the following command. 
```bash
python train_trl.py
```
