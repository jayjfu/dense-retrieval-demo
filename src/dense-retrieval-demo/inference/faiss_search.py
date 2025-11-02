import argparse
import os
import torch
from models import BertForSequenceClassification
from models import BertConfig, BertModel
from utils import BertTokenizer
import json
import pandas as pd
from tqdm import tqdm
import faiss


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='encoding for inference')
parser.add_argument('--model_config', default='../data/config.json', type=str)
parser.add_argument('--model_weights', default='../checkpoints/bert_msmarco_model.pt', type=str)
parser.add_argument('--vocab_file', default='../data/vocab.txt', type=str)
parser.add_argument('--max_length', default=128, type=int)
parser.add_argument('--file_path', default='../../../benchmarks/msmarco-passage-ranking/data/', type=str)
parser.add_argument('--file_name', default='queries.dev.tsv', type=str)
parser.add_argument('--qrels_dev_file', default='qrels.dev.tsv', type=str)
parser.add_argument('--index_path', default='../../../benchmarks/msmarco-passage-ranking/index/', type=str)
parser.add_argument('--index_file', default='passages_100k_index.faiss', type=str)
parser.add_argument('--eval_path', default='../../../benchmarks/msmarco-passage-ranking/eval/', type=str)
parser.add_argument('--prediction_file', default='bert.ranking_results.dev.tsv', type=str)
parser.add_argument('--top_k', default=10, type=str)
args = parser.parse_args()

def main():
    qrels_path = os.path.join(str(SCRIPT_DIR), args.file_path, args.qrels_dev_file)
    query_path = os.path.join(str(SCRIPT_DIR), args.file_path, args.file_name)

    qid2pid_df = pd.read_csv(qrels_path, sep="\t", header=None, names=['qid', 'zero', 'pid', 'label'])
    qid2query_df = pd.read_csv(query_path, sep="\t", header=None, names=['qid', 'query_text'])
    merged_df = qid2pid_df.merge(qid2query_df, on="qid", how="left")
    dataset = merged_df[['qid', 'query_text']].to_dict(orient='records')

    tokenizer = BertTokenizer(os.path.join(SCRIPT_DIR, args.vocab_file))

    model_config = BertConfig(**json.load(open(os.path.join(SCRIPT_DIR, args.model_config))))
    model = BertForSequenceClassification(BertModel(model_config))
    checkpoint = torch.load(os.path.join(str(SCRIPT_DIR), args.model_weights), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    index_path = os.path.join(SCRIPT_DIR, args.index_path, args.index_file)
    index = faiss.read_index(index_path)

    output_dir = os.path.join(str(SCRIPT_DIR), args.eval_path)
    os.makedirs(output_dir, exist_ok=True)
    prediction_file = os.path.join(output_dir, args.prediction_file)

    with open(prediction_file, 'w') as f:
        for row in tqdm(dataset):
            qid, query_text = row['qid'], row['query_text']

            input_ids, attention_mask, _ = tokenizer.encode(query_text, max_length=args.max_length)
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            attention_mask = torch.tensor(attention_mask).unsqueeze(0)
            with torch.no_grad():
                outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)

            # Average pooling
            last_hidden_state = outputs[0]
            query_emb = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(1)
            query_emb = query_emb / attention_mask.sum(1, keepdim=True)

            D, I = index.search(query_emb.cpu().numpy(), k=args.top_k)

            for rank, pid in enumerate(I[0], start=1):
                f.write(f"{qid}\t{pid}\t{rank}\n")

if __name__ == "__main__":
    main()