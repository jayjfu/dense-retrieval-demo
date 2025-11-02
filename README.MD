
# Dense Retrieval Demo

---
### This demo showcases a simple workflow for **training dense retrieval models** and **performing inference** with them.

*(Work in Progress: This repository is currently under active development.)*

## Introduction

---
This repository is compatible with datasets in the format like <u>MS MARCO Passage Ranking</u> dataset. We use a **standard BERT model** for case study.

## Data Preparation

---
### MS MARCO Passage Ranking
```bash
cd benchmarks/msmarco-passage-ranking
bash get_dataset.sh
```

### Pretrained BERT weights
```bash
bash get_pretrained.sh
```

## Training

---
### Hugging Face BERT
Train a standard BERT classification model using Hugging Face:
```bash
# Training w/ HF
python ./src/dense-retrieval-demo/hf_train.py
```

### Custom BERT
Train a custom BERT classification model:
```bash
# Training a custom 
python ../src/dense-retrieval-demo/train.py
```

## Inference

---
### Retrieval + Rerank
End-to-end retrieval & rerank:
```bash
# Indexing w/ HF
python ./src/dense-retrieval-demo/inference/hf_encoding.py

# FAISS search w/ HF
python ./src/dense-retrieval-demo/inference/hf_faiss_search.py
```
```bash
# Indexing (custom)
python -m inference.encoding

# FAISS Search (custom)
python -m inference.faiss_search
```

### Rerank Only
For rerank only,

*(Work in Progress)*
```bash
# To be implemented
```

## Evaluation

---
Results on the MS MARCO passage ranking task:

*(Work in Progress)*

| Model   | Retrieval + Rerank | Rerank |
|---------|--------------------|--------|
| Model_1 | -                  | -      |
| Model_2 | -                  | -      |

## Scripts:

---

- Training: scripts/training_pipeline.sh
- Inference: scripts/hf_inference_pipeline.sh, scripts/inference_pipeline.sh
- Evaluation script: scripts/evaluation.sh

## Reference & Acknowledgement:

---
- Pretrained BERT weights: [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased/tree/main) 
- BERT tokenizer from Google: [tokenization.py](https://github.com/google-research/bert/blob/master/tokenization.py)
