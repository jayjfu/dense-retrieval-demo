import argparse
import os
import torch
from dataset import PassageDataset
from models import BertForSequenceClassification, BertConfig, BertModel
from utils import BertTokenizer
import numpy as np
import json
import faiss
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="encoding for inference")
parser.add_argument('--model_config', default="../data/pretrained_bert/config.json", type=str)
parser.add_argument('--model_weights', default="../checkpoints/custom-bert-msmarco/ckpt_step1240000.pt", type=str)
parser.add_argument('--file_path', default="../../../benchmarks/msmarco-passage-ranking/data/", type=str)
parser.add_argument('--file_name', default="collection.tsv", type=str)
parser.add_argument('--vocab_file', default="../data/pretrained_bert/vocab.txt", type=str)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--max_length', default=128, type=int)
parser.add_argument('--index_path', default="../../../benchmarks/msmarco-passage-ranking/index/", type=str)
parser.add_argument('--index_file', default="passages_index.faiss", type=str)
args = parser.parse_args()

def main():
    passage_path = os.path.join(str(SCRIPT_DIR), args.file_path, args.file_name)
    passages = PassageDataset(passage_path, limit=None).data

    tokenizer = BertTokenizer(os.path.join(SCRIPT_DIR, args.vocab_file))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_config = BertConfig(**json.load(open(os.path.join(SCRIPT_DIR, args.model_config))))
    model = BertForSequenceClassification(BertModel(model_config))
    checkpoint = torch.load(os.path.join(str(SCRIPT_DIR), args.model_weights), map_location=device) # 'cpu'
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)

    def gen_embeddings(passages, batch_size):
        # all_embeddings = []
        emb_dim = 768
        all_embeddings = np.memmap('embeddings.npy', dtype='float32', mode='w+', shape=(len(passages), emb_dim))

        total_batches = (len(passages) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc='Encoding')

        for start_idx in range(0, len(passages), batch_size):
            batch = passages[start_idx:start_idx+batch_size]

            input_ids, attention_mask = [], []
            for p in batch:
                p_input_ids, p_attention_mask, _ = tokenizer.encode(p, max_length=args.max_length)
                input_ids.append(p_input_ids)
                attention_mask.append(p_attention_mask)

            input_ids = torch.tensor(input_ids).to(device)
            attention_mask = torch.tensor(attention_mask).to(device)
            with torch.no_grad():
                outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)

            # Average pooling
            last_hidden_state = outputs[0]
            embeddings = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(1)
            embeddings = embeddings / attention_mask.sum(1, keepdim=True)
            embeddings = faiss.normalize_L2(embeddings)  # l2 norm

            # all_embeddings.append(embeddings.cpu().numpy())
            end_idx = min(start_idx + batch_size, len(passages))
            all_embeddings[start_idx:end_idx] = embeddings.cpu().numpy()
            all_embeddings.flush()

            pbar.update(1)

        return all_embeddings

    embeddings = gen_embeddings(passages, batch_size=args.batch_size)

    dim = len(embeddings[0])
    index = faiss.index_factory(dim, "IVF4096,Flat", faiss.METRIC_INNER_PRODUCT)  # PQ32

    bs = 5_000_000
    train_size = min(bs, len(embeddings))
    train_embeddings = np.vstack(embeddings[:train_size])
    index.train(train_embeddings)

    for i in range(0, len(embeddings), bs):
        batch = np.vstack(embeddings[i:i + bs])
        index.add(batch)

    os.remove('embeddings.npy')

    os.makedirs(os.path.join(SCRIPT_DIR, args.index_path), exist_ok=True)
    faiss.write_index(index, os.path.join(SCRIPT_DIR, args.index_path, args.index_file))

if __name__ == "__main__":
    main()