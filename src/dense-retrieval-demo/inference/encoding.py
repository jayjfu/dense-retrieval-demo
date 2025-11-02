import argparse
import os
import torch
from dataset import PassageDataset
from models import BertForSequenceClassification
from models import BertConfig, BertModel
from utils import BertTokenizer
import numpy as np
import json
import faiss


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='encoding for inference')
parser.add_argument('--model_config', default='../data/config.json', type=str)
parser.add_argument('--model_weights', default='../checkpoints/bert_msmarco_model.pt', type=str)
parser.add_argument('--file_path', default='../../../benchmarks/msmarco-passage-ranking/data/', type=str)
parser.add_argument('--file_name', default='collection.tsv', type=str)
parser.add_argument('--vocab_file', default='../data/vocab.txt', type=str)
parser.add_argument('--max_length', default=128, type=int)
parser.add_argument('--index_path', default='../../../benchmarks/msmarco-passage-ranking/index/', type=str)
parser.add_argument('--index_file', default='passages_100k_index.faiss', type=str)
args = parser.parse_args()

def main():
    passage_path = os.path.join(str(SCRIPT_DIR), args.file_path, args.file_name)
    passages = PassageDataset(passage_path, limit=1_000).data

    tokenizer = BertTokenizer(os.path.join(SCRIPT_DIR, args.vocab_file))

    model_config = BertConfig(**json.load(open(os.path.join(SCRIPT_DIR, args.model_config))))
    model = BertForSequenceClassification(BertModel(model_config))
    checkpoint = torch.load(os.path.join(str(SCRIPT_DIR), args.model_weights), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    def gen_embeddings(passages, batch_size=64):
        all_embeddings = []

        for i in range(0, len(passages), batch_size):
            batch = passages[i:i+batch_size]

            input_ids, attention_mask = [], []
            for p in batch:
                p_input_ids, p_attention_mask, _ = tokenizer.encode(p, max_length=args.max_length)
                input_ids.append(p_input_ids)
                attention_mask.append(p_attention_mask)

            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)
            with torch.no_grad():
                outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)

            # Average pooling
            last_hidden_state = outputs[0]
            embeddings = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(1)
            embeddings = embeddings / attention_mask.sum(1, keepdim=True)

            all_embeddings.append(embeddings.cpu().numpy())

        return all_embeddings

    embeddings = gen_embeddings(passages, batch_size=64)
    embeddings = np.vstack(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(os.path.join(SCRIPT_DIR, args.index_path), exist_ok=True)
    faiss.write_index(index, os.path.join(SCRIPT_DIR, args.index_path, args.index_file))

if __name__ == "__main__":
    main()