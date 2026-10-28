# Improving Agentless: Rank-Aware Localization and Evidence-Guided Patch Selection

This repository contains our research project extending **Agentless**, an agentless approach for automated program repair on SWE-bench.

The project investigates whether improvements to Agentless's **fault localization** and **patch selection** stages can improve the reliability of automated program repair.

Our work is based on the original [Agentless](https://github.com/OpenAutoCoder/Agentless) implementation and the SWE-bench Lite benchmark.

---

## 📌 About the Project

**Agentless** is an agentless approach to automated software engineering that follows three main phases:

1. **Localization** — identify the files, code elements, and edit locations relevant to an issue.
2. **Repair** — generate multiple candidate patches.
3. **Patch Validation and Selection** — validate candidate patches and select a final patch.

The original Agentless localization process uses both **LLM-based localization** and **embedding-based retrieval** and combines their results to identify suspicious files.

This project investigates two modifications to this pipeline:

### 1. Rank-Aware Fault Localization

The original Agentless file-level localization combines LLM and embedding results using occurrence-based voting.

We replace this voting mechanism with **Reciprocal Rank Fusion (RRF)** so that the rank of a file within each retrieval source is taken into account.

### 2. Evidence-Guided Patch Re-Ranking

The original Agentless pipeline generates multiple candidate patches and uses validation results to assist in selecting the final patch.

We investigate an **evidence-guided patch re-ranking approach** that combines test results with semantic evaluation of candidate patches to improve patch selection.

---

## 🎯 Research Questions

### RQ1 — Can rank-aware fusion improve Agentless's file-level localization?

We examine whether replacing Agentless's current voting approach with **Reciprocal Rank Fusion (RRF)** can better combine the ranked results from LLM and embedding-based localization and identify the correct files.

### RQ2 — Can evidence-guided patch re-ranking improve patch selection?

We examine whether using **test results and semantic evaluation of the patches** to rank validated candidates can select the correct patch more reliably than Agentless's current patch selection approach.

---

## 🔬 Experimental Pipeline

Our modified Agentless pipeline is structured as follows:

```text
                    SWE-bench Issue
                          │
                          ▼
              ┌─────────────────────┐
              │ File-Level          │
              │ Localization        │
              └──────────┬──────────┘
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       LLM Localization       Embedding Retrieval
             │                       │
             └───────────┬───────────┘
                         ▼
              ┌─────────────────────┐
              │ Reciprocal Rank     │
              │ Fusion (RRF)        │
              └──────────┬──────────┘
                         │
                         ▼
              Related Element
              Localization
                         │
                         ▼
              Fine-Grained Edit
                 Localization
                         │
                         ▼
              Candidate Patch
                 Generation
                         │
                         ▼
              Patch Validation
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       Test Results          Semantic Evaluation
             │                       │
             └───────────┬───────────┘
                         ▼
              Evidence-Guided
              Patch Re-Ranking
                         │
                         ▼
                  Final Patch
```

---

## ⚙️ Setup

Create the environment:

```bash
git clone <your-repository-url>
cd <your-repository-name>

conda create -n agentless-project python=3.11
conda activate agentless-project

pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

Set the OpenAI API key required by the Agentless pipeline:

```bash
export OPENAI_API_KEY={your_key}
```

---

## 🧪 Running Agentless

The localization pipeline follows the original Agentless workflow:

```text
LLM File Localization
        ↓
Embedding Retrieval
        ↓
File-Level Fusion
        ↓
Related Element Localization
        ↓
Fine-Grained Edit Localization
        ↓
Patch Generation
        ↓
Patch Validation
        ↓
Patch Selection
```

---

## 📚 Dataset

Experiments are conducted using **SWE-bench Lite**.

The benchmark contains real-world software engineering issues from open-source repositories. The reference patches associated with the benchmark are used to determine the ground-truth modified files for localization evaluation.

The original Agentless repository also provides classifications and artifacts associated with SWE-bench Lite.

