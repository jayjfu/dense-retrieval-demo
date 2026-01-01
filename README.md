
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
python tokenizer.py
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

## Inference

### Retrieval + Rerank
End-to-end retrieval & rerank:
```bash
# Indexing w/ HF
python ./src/dense-retrieval-demo/inference/hf_encoding.py

# FAISS search w/ HF
python ./src/dense-retrieval-demo/inference/hf_faiss_search.py
```

Use custom classification model: 
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
# TODO
```

## Evaluation

We use the official evaluation script here:
```bash
python ./benchmarks/msmarco-passage-ranking/eval/ms_marco_eval.py $reference_file $candidate_file
```

MRR@10 (Dev) results of  on the MS MARCO passage ranking task: *(Work in Progress)*

| Model                                | Retrieval + Rerank | Rerank Only |
|--------------------------------------|--------------------|-------------|
| BERT_base (classifier head only)     | 0.005              | -           |
| BERT_base (encoder and classifier)   | -                  | -           |
| custom_BERT (classifier head only)   | 0.000              | -           |
| custom_BERT (encoder and classifier) | -                  | -           |

Note: Here, we use the small training set `triples.train.small.tar.gz` and set max_length=128 to improve training performance.

## Scripts:

- Training: `scripts/training_pipeline.sh`
- Inference: `scripts/hf_inference_pipeline.sh`, `scripts/inference_pipeline.sh`
- Evaluation `script: scripts/evaluation.sh`

## Reference & Acknowledgement:

- Pretrained BERT weights: [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased/tree/main) 
- BERT tokenizer from Google: [tokenization.py](https://github.com/google-research/bert/blob/master/tokenization.py)
