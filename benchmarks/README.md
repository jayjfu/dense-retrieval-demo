# MS MARCO Passage Ranking Task

## Data Preparation

- Step1: download dataset:
```bash
cd msmarco-passage-ranking
bash get_dataset.sh
```

- Step2: tokenization: 
```bash
python pre_tokenizer.py --max_length 128 # add --custom_tokenizer to use custom tokenizer
```

## Evaluation

We use the official evaluation script here:
```bash
# https://github.com/microsoft/MSMARCO-Passage-Ranking/blob/master/ms_marco_eval.py
python msmarco-passage-ranking/eval/ms_marco_eval.py $reference_file $candiadate_file
```
