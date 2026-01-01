import argparse
import os
import torch
import numpy as np
from networkx.algorithms.planar_drawing import triangulate_embedding
from transformers import AutoTokenizer, AutoModel
import datasets
import faiss


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="encoding for inference")
parser.add_argument('--saved_model', default="../checkpoints/bert-msmarco/checkpoint-1243152", type=str)
parser.add_argument('--file_path', default="../../../benchmarks/msmarco-passage-ranking/data/", type=str)
parser.add_argument('--file_name', default="collection.tsv", type=str)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--max_length', default=128, type=int)
parser.add_argument('--index_path', default="../../../benchmarks/msmarco-passage-ranking/index/", type=str)
parser.add_argument('--index_file', default="hf_passages_index.faiss", type=str)
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

    dataset = dataset['train'].map(tokenize_passage, batched=True, num_proc=8, remove_columns=dataset['train'].column_names)
    dataset.set_format(type='torch')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoModel.from_pretrained(os.path.join(str(SCRIPT_DIR), args.saved_model))
    model.eval()
    model.to(device)

    def gen_embeddings(batch):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Average pooling
        embeddings = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1)
        embeddings = embeddings / attention_mask.sum(1, keepdim=True)

        return {"embeddings": embeddings.cpu().numpy()}

    emb_dataset = dataset.map(gen_embeddings, batched=True, batch_size=args.batch_size)

    dim = len(emb_dataset[0]['embeddings'])
    index = faiss.index_factory(dim, "IVF4096,PQ32")

    bs = 5_000_000
    train_size = min(bs, len(emb_dataset))
    train_embeddings = np.vstack(emb_dataset[:train_size]['embeddings'])
    index.train(train_embeddings)

    for i in range(0, len(emb_dataset), bs):
        batch = np.vstack(emb_dataset[i:i + bs]['embeddings'])
        index.add(batch)

    os.makedirs(os.path.join(SCRIPT_DIR, args.index_path), exist_ok=True)
    faiss.write_index(index, os.path.join(SCRIPT_DIR, args.index_path, args.index_file))

if __name__ == "__main__":
    main()