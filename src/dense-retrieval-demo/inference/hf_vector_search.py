import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import datasets
import pandas as pd
import faiss
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="encoding for inference")
parser.add_argument('--saved_model', default="../checkpoints/bert-msmarco/checkpoint-1243152", type=str)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--max_length', default=128, type=int)
parser.add_argument('--file_path', default="../../../benchmarks/msmarco-passage-ranking/data/", type=str)
parser.add_argument('--file_name', default="queries.dev.tsv", type=str)
parser.add_argument('--qrels_dev_file', default="qrels.dev.tsv", type=str)
parser.add_argument('--index_path', default="../../../benchmarks/msmarco-passage-ranking/index/", type=str)
parser.add_argument('--index_file', default="hf_passages_index.faiss", type=str)
parser.add_argument('--eval_path', default="../../../benchmarks/msmarco-passage-ranking/eval/", type=str)
parser.add_argument('--prediction_file', default="hf_bert.ranking_results.dev.tsv", type=str)
parser.add_argument('--top_k', default=10, type=int)
parser.add_argument('--index_nprobe', default=10, type=int)
parser.add_argument('--model_type', default="cross-encoder", choices=["cross-encoder", "bi-encoder"])
parser.add_argument('--reranking_only', action='store_true')
parser.add_argument('--bm25_retrieval_file_name', default='top1000.dev')
args = parser.parse_args()

def score_cross(model, tokenizer, query, passages, device, batch_size, max_length):
    pairs = [[query, p] for p in passages]
    scores = []

    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + batch_size]
        enc = tokenizer(batch_pairs, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)

        with torch.no_grad():
            out = model(**enc)
            batch_scores = F.softmax(out.logits)[:, 1].cpu().numpy()

        scores.extend(batch_scores)

    return np.array(scores)

def encode_texts(model, tokenizer, device, texts, batch_size=args.batch_size, max_length=args.max_length):
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            # Average pooling
            emb = (outputs.last_hidden_state * encoded['attention_mask'].unsqueeze(-1)).sum(1)
            emb = emb / encoded['attention_mask'].sum(1, keepdim=True)
            embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings)

def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def main():
    qrels_path = os.path.join(str(SCRIPT_DIR), args.file_path, args.qrels_dev_file)
    query_path = os.path.join(str(SCRIPT_DIR), args.file_path, args.file_name)

    qid2pid_df = pd.read_csv(qrels_path, sep='\t', header=None, names=['qid', 'zero', 'pid', 'label'])
    qid2query_df = pd.read_csv(query_path, sep='\t', header=None, names=['qid', 'query_text'])
    merged_df = qid2pid_df.merge(qid2query_df, on='qid', how='left')
    dataset = datasets.Dataset.from_pandas(merged_df)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    model_path = os.path.join(str(SCRIPT_DIR), args.saved_model)

    if args.model_type == "cross-encoder":
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("Loaded as cross-encoder (AutoModelForSequenceClassification)")
    elif args.model_type == "bi-encoder":
        model = AutoModel.from_pretrained(model_path)
        print("Loaded as bi-encoder (AutoModel)")
    else:
        raise Exception("Unknown model type")

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if args.reranking_only:
        top1000_path = os.path.join(str(SCRIPT_DIR), args.file_path, args.bm25_retrieval_file_name)
        top1000_df = pd.read_csv(top1000_path, sep='\t', header=None, names=['qid', 'pid', 'query_text', 'passage_text'])
        passages_dict = top1000_df.groupby('qid')['passage_text'].apply(list).to_dict()
        pids_dict = top1000_df.groupby('qid')['pid'].apply(list).to_dict()
    else:
        if args.model_type == "cross-encoder":
            raise ValueError("Cross-encoder can only be used in --reranking_only mode")

        index_path = os.path.join(SCRIPT_DIR, args.index_path, args.index_file)
        index = faiss.read_index(index_path)
        index.nprobe = args.index_nprobe  # search more clusters

    output_dir = os.path.join(str(SCRIPT_DIR), args.eval_path)
    os.makedirs(output_dir, exist_ok=True)
    prediction_file = os.path.join(output_dir, args.prediction_file)

    with open(prediction_file, 'w') as f:
        for row in tqdm(dataset):
            qid, query_text = row['qid'], row['query_text']

            if args.reranking_only:
                candidate_passages = passages_dict.get(qid, [])
                candidate_pids = pids_dict.get(qid, [])

                if not candidate_pids:
                    continue

                if args.model_type == "cross-encoder":
                    scores = score_cross(model, tokenizer, query_text, candidate_passages, device, args.batch_size, args.max_length)
                else:
                    query_emb = encode_texts(model, tokenizer, device, [query_text], 1, args.max_length)
                    passage_embs = encode_texts(model, tokenizer, device, list(candidate_passages), args.batch_size, args.max_length)

                    query_emb = l2_normalize(query_emb)
                    passage_embs = l2_normalize(passage_embs)
                    scores = np.dot(passage_embs, query_emb.T).squeeze()  # cosine sim

                # rank
                ranked_idx = scores.argsort()[::-1][:args.top_k]
                for rank, idx in enumerate(ranked_idx, 1):
                    pid = candidate_pids[idx]
                    f.write(f"{qid}\t{pid}\t{rank}\n")
            else:
                tokenized_query = tokenizer(query_text, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt')
                input_ids = tokenized_query['input_ids'].to(device)
                attention_mask = tokenized_query['attention_mask'].to(device)

                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                # Average pooling
                query_emb = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1)
                query_emb = query_emb / attention_mask.sum(1, keepdim=True)
                query_emb = l2_normalize(query_emb)  # l2 norm

                D, I = index.search(query_emb.cpu().numpy(), k=args.top_k)

                for rank, pid in enumerate(I[0], start=1):
                    f.write(f"{qid}\t{pid}\t{rank}\n")

if __name__ == "__main__":
    main()