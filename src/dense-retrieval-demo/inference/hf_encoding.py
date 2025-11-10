import argparse
import os
import torch
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModel
import datasets
import faiss


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="encoding for inference")
parser.add_argument('--saved_model', default="../checkpoints/bert-msmarco/checkpoint-5000", type=str)
parser.add_argument('--file_path', default="../../../benchmarks/msmarco-passage-ranking/data/", type=str)
parser.add_argument('--file_name', default="collection.tsv", type=str)
parser.add_argument('--max_length', default=128, type=int)
parser.add_argument('--index_path', default="../../../benchmarks/msmarco-passage-ranking/index/", type=str)
parser.add_argument('--index_file', default="hf_passages_100k_index.faiss", type=str)
args = parser.parse_args()

def main():
    dataset = datasets.load_dataset(
        'csv',
        data_files=os.path.join(str(SCRIPT_DIR), args.file_path, args.file_name),
        delimiter='\t',
        column_names=['pid', 'passage_text'],
    )

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    def tokenize_passage(example):
        return tokenizer(example['passage_text'], padding=True, truncation=True, max_length=args.max_length)
    dataset = dataset['train'].select(range(100_000)).map(tokenize_passage, batched=True, remove_columns=dataset['train'].column_names)
    dataset = dataset.with_format(type='torch')

    model = AutoModel.from_pretrained(os.path.join(str(SCRIPT_DIR), args.saved_model))
    model.eval()

    def gen_embeddings(batch):
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

        # Average pooling
        embeddings = (outputs.last_hidden_state * batch['attention_mask'].unsqueeze(-1)).sum(1)
        embeddings = embeddings / batch['attention_mask'].sum(1, keepdim=True)

        return {"embeddings": embeddings.cpu().numpy()}
    emb_dataset = dataset.map(gen_embeddings, batched=True, batch_size=64)

    embeddings = np.vstack(emb_dataset['embeddings'])
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(os.path.join(SCRIPT_DIR, args.index_path), exist_ok=True)
    faiss.write_index(index, os.path.join(SCRIPT_DIR, args.index_path, args.index_file))

if __name__ == "__main__":
    main()