#  Conversation Evaluation Task

This repository implements a **scalable, facet-based conversation evaluation system** that generates **fine-grained psychological and linguistic scores** for conversational texts across **300+ behavioral, emotional, cognitive, and social facets** using open-weight LLMs.

Built under real-world **production constraints** (≤16B models, no one-shot prompting), this system supports **turn-level evaluation** and **automated benchmarking** for conversational agents in domains like customer support, coaching, therapy, and more.

---

## 🎯 Project Objective

The goal is to **automatically evaluate each sentence/turn** in a conversation across a rich set of facets — such as **empathy, assertiveness, risk-taking, contentment, emotional depth**, and many others — by generating scores using **open models**, **weak supervision**, and a **regression-based scoring model**.

This enables:

- ✅ Explainable LLM evaluation  
- ✅ Faceted dialog quality benchmarking  
- ✅ Behavioral and linguistic analysis at scale  

---

## 🧩 Workflow Overview

### ✅ Step 1: Facet Preprocessing

- Raw facet names are cleaned (colons/symbols removed, duplicates dropped).
- Reformatted for consistent and safe prompt injection.
- Chunked into batches of 10 facets per call to fit LLM context length constraints.

---

### 🗣️ Step 2: Synthetic Sentence Generation

- For each facet, **10 example sentences** are generated using:
  > [`mistralai/Mistral-7B-Instruct-v0.1`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)

- These sentences reflect **semantic alignment** with the facet and simulate natural conversational turns.
- Used as **training examples** for the scoring model.

---

### 📊 Step 3:  Label Generation (Scoring)

- Each sentence is evaluated for its alignment with each facet using:
  > [`meta-llama/Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

- Prompts elicit a **1–5 scale rating** for each facet:  
  `1 = Not related` → `5 = Very strongly related`

- **Regex parsing** is used to extract scores from model output.
- If the model fails to return valid scores, we assign a **default neutral score of 3**.

---

### 📈 Step 4: Multi-Output Regression Model Training

- The generated (sentence, facet, score) triplets are used to train a **multi-output regression model**.
- Sentence embeddings are extracted using a transformer-based encoder (e.g., `sentence-transformers`).
- The trained model can **predict scores for new sentences without needing an LLM**, enabling fast and scalable inference.

---

## 💻 Running the Pipeline

### 📦 Installation

```bash
git clone https://github.com/whoami-saloni/Conversation-Evaluation-Task.git
cd Conversation-Evaluation-Task
pip install -r requirements.txt
