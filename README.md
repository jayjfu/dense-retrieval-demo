
# Dense Retrieval Demo

### This demo showcases a simple workflow for **training dense retrieval models** and **performing inference** with them.

*(Work in Progress: This repository is currently under active development.)*

## Introduction

This repository is compatible with datasets in the format like <u>MS MARCO Passage Ranking</u> dataset. We use a **standard BERT model** for case study.

## Project Structure
```
dense-retrieval-demo
├── README.MD
├── benchmarks
│   ├── README.md
│   └── msmarco-passage-ranking
│       ├── eval
│       │   └── ms_marco_eval.py
│       ├── get_dataset.sh
│       └── tokenizer.py
├── requirements.txt
├── scripts
│   ├── evaluation.sh
│   ├── hf_inference_pipeline.sh
│   ├── inference_pipeline.sh
│   └── training_pipeline.sh
└── src
    └── dense-retrieval-demo
        ├── dataset
        │   ├── __init__.py
        │   ├── collator.py
        │   └── dataset.py
        ├── get_pretrained.sh
        ├── hf_train.py
        ├── inference
        │   ├── __init__.py
        │   ├── encoding.py
        │   ├── faiss_search.py
        │   ├── hf_encoding.py
        │   └── hf_faiss_search.py
        ├── models
        │   ├── __init__.py
        │   ├── bert_backbone.py
        │   └── bert_classifier.py
        ├── train.py
        └── utils
            ├── __init__.py
            └── bert_tokenization.py
```

## Data Preparation

### MS MARCO Passage Ranking
Download dataset:
```bash
cd benchmarks/msmarco-passage-ranking
bash get_dataset.sh
```

Data preprocessing (tokenization):
```bash
python tokenizer.py \
  --max_length 128 \
  --num_processes 8 \
  --mp_chunk_size 100_000
```

### Pretrained BERT weights
Download pretrained model:
```bash
bash get_pretrained.sh
```

## Training

### Hugging Face BERT
Train a standard BERT classification model using Hugging Face:
```bash
# Training w/ HF
python ./src/dense-retrieval-demo/hf_train.py \
  --batch_size 128 \
  --lr 5e-6 \
  --num_epochs 2 \
  --save_steps 20_000 \
  --no_resume
```

### Custom BERT
Train a custom BERT classification model:
```bash
# Training a custom 
python ./src/dense-retrieval-demo/train.py \
  --batch_size 128 \
  --lr 2e-5 \
  --num_epochs 2 \
  --logging_steps 2_000 \
  --save_steps 20_000 \
  --no_resume
```

Note: Bi-Encoder or Cross-Encoder?? + full fine-tuning or classifier head tuning??

## Inference

### Retrieval + Rerank
End-to-end retrieval & rerank:
```bash
# Indexing w/ HF
python ./src/dense-retrieval-demo/inference/hf_encoding.py \
  --batch_size 1024 \
  --max_length 128

# FAISS search w/ HF
python ./src/dense-retrieval-demo/inference/hf_faiss_search.py \
  --batch_size 1024 \
  --max_length 128 \
  --top_k 10 \
  --index_nprobe 8
```

Use custom classification model: 
```bash
# Indexing (custom)
python -m inference.encoding \
  --batch_size 1024 \
  --max_length 128

# FAISS Search (custom)
python -m inference.faiss_search \
  --batch_size 1024 \
  --max_length 128 \
  --tok_k 10 \
  --index_nprobe 8
```

Note: Bi-Encoder or Cross-Encoder??

### Reranking Only
To rerank the official top-1000 results produced by BM25, add `--reranking_only` flag during faiss search:

```bash
# FAISS search (reranking only)
python ./src/dense-retrieval-demo/inference/hf_faiss_search.py --reranking_only
# Or
python -m inference.faiss_search --reranking_only
```

## Evaluation

We use the official evaluation script here:
```bash
python ./benchmarks/msmarco-passage-ranking/eval/ms_marco_eval.py $reference_file $candidate_file
```

MRR@10 (Dev) results of  on the MS MARCO passage ranking task: *(Work in Progress)*

- Retrieval + Reranking:

| Backbone    | Fine-tuning Scope  | Encoder Type  | Score |
|-------------|--------------------|---------------|-------|
| BERT_base   | Full fine-tuning   | Bi-encoder    | -     |
| custom_BERT | Full fine-tuning   | Bi-encoder    | -     |

- Reranking Only:

| Backbone    | Fine-tuning Scope  | Encoder Type  | Score |
|-------------|--------------------|---------------|-------|
| BERT_base   | Classifier head    | Cross-encoder | -     |
| BERT_base   | Full fine-tuning   | Cross-encoder | -     |
| BERT_base   | Full fine-tuning   | Bi-encoder    | -     |
| custom_BERT | Classifier head    | Cross-encoder | -     |
| custom_BERT | Full fine-tuning   | Cross-encoder | -     |
| custom_BERT | Full fine-tuning   | Bi-encoder    | -     |

Note: We use the smaller training set (`triples.train.small.tar.gz`) to speed up training.

## Scripts:

- Training: `scripts/training_pipeline.sh`
- Inference: `scripts/hf_inference_pipeline.sh`, `scripts/inference_pipeline.sh`
- Evaluation `script: scripts/evaluation.sh`

## Reference & Acknowledgement:

- Pretrained BERT weights: [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased/tree/main) 
- BERT tokenizer from Google: [tokenization.py](https://github.com/google-research/bert/blob/master/tokenization.py)
