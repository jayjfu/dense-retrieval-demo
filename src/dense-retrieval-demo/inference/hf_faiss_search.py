import argparse
import os
import torch
from transformers import AutoTokenizer
from transformers import AutoModel
import datasets
import pandas as pd
from tqdm import tqdm
import faiss


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='encoding for inference')
parser.add_argument('--saved_model', default='../checkpoints/bert-msmarco/checkpoint-5000', type=str)
parser.add_argument('--max_length', default=128, type=int)
parser.add_argument('--file_path', default='../../../benchmarks/msmarco-passage-ranking/data/', type=str)
parser.add_argument('--file_name', default='queries.dev.tsv', type=str)
parser.add_argument('--qrels_dev_file', default='qrels.dev.tsv', type=str)
parser.add_argument('--index_path', default='../../../benchmarks/msmarco-passage-ranking/index/', type=str)
parser.add_argument('--index_file', default='hf_passages_100k_index.faiss', type=str)
parser.add_argument('--eval_path', default='../../../benchmarks/msmarco-passage-ranking/eval/', type=str)
parser.add_argument('--prediction_file', default='hf_bert.ranking_results.dev.tsv', type=str)
parser.add_argument('--top_k', default=10, type=str)
args = parser.parse_args()

def main():
    qrels_path = os.path.join(str(SCRIPT_DIR), args.file_path, args.qrels_dev_file)
    query_path = os.path.join(str(SCRIPT_DIR), args.file_path, args.file_name)

    qid2pid_df = pd.read_csv(qrels_path, sep="\t", header=None, names=['qid', 'zero', 'pid', 'label'])
    qid2query_df = pd.read_csv(query_path, sep="\t", header=None, names=['qid', 'query_text'])
    merged_df = qid2pid_df.merge(qid2query_df, on="qid", how="left")
    dataset = datasets.Dataset.from_pandas(merged_df)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    model = AutoModel.from_pretrained(os.path.join(str(SCRIPT_DIR), args.saved_model))
    model.eval()

    index_path = os.path.join(SCRIPT_DIR, args.index_path, args.index_file)
    index = faiss.read_index(index_path)

    output_dir = os.path.join(str(SCRIPT_DIR), args.eval_path)
    os.makedirs(output_dir, exist_ok=True)
    prediction_file = os.path.join(output_dir, args.prediction_file)

    with open(prediction_file, 'w') as f:
        for row in tqdm(dataset):
            qid, query_text = row['qid'], row['query_text']

            tokenized_query = tokenizer(query_text, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt')
            with torch.no_grad():
                outputs = model(input_ids=tokenized_query['input_ids'], attention_mask=tokenized_query['attention_mask'])

            # Average pooling
            query_emb = (outputs.last_hidden_state * tokenized_query['attention_mask'].unsqueeze(-1)).sum(1)
            query_emb = query_emb / tokenized_query['attention_mask'].sum(1, keepdim=True)

            D, I = index.search(query_emb.cpu().numpy(), k=args.top_k)

            for rank, pid in enumerate(I[0], start=1):
                f.write(f"{qid}\t{pid}\t{rank}\n")

if __name__ == "__main__":
    main()