import argparse
import os
import torch
from dataset import TokenizedTextPairDataset, pad_to_max_length
from models import BertForSequenceClassification, BertConfig, BertModel
from torch.utils.data import DataLoader
import json
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='train models for neural search task')
parser.add_argument('--data_path', default='../../benchmarks/msmarco-passage-ranking/data/processed/train/ms_train.jsonl', type=str)
parser.add_argument('--model_config', default='./data/config.json', type=str)
parser.add_argument('--model_weights', default='./data/pytorch_model.bin', type=str)
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--epochs', default=3, type=int)
parser.add_argument('--val_interval', default=5_00, type=int)
parser.add_argument('--output_dir', default='./checkpoints', type=str)
parser.add_argument('--saved_model', default='bert_msmarco_model.pt', type=str)
args = parser.parse_args()

def train(args):
    data_path = os.path.join(SCRIPT_DIR, args.data_path)
    dataset = TokenizedTextPairDataset(data_path, limit=1_000).data
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=pad_to_max_length)

    model_config = BertConfig(**json.load(open(os.path.join(SCRIPT_DIR, args.model_config))))
    model = BertForSequenceClassification(BertModel(model_config))

    model_weights = os.path.join(str(SCRIPT_DIR), args.model_weights)
    state_dict = torch.load(model_weights, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    step = 0
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        total_loss = 0

        for batch in tqdm(dataloader, desc='Training'):
            step += 1

            model.train()
            input_ids, labels = batch['input_ids'], batch['labels']

            outputs = model(input_ids, labels)
            loss = outputs['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Validation and save
            if step % args.val_interval == 0:
                model.eval()
                val_loss = 0
                for batch in dataloader:
                    input_ids, labels = batch['input_ids'], batch['labels']

                    outputs = model(input_ids, labels)
                    loss = outputs['loss']
                    val_loss += loss.item()

                val_loss = val_loss / len(dataloader)
                print(f'Epoch {epoch + 1} | Val Loss: {val_loss:.4f}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    os.makedirs(os.path.join(SCRIPT_DIR, args.output_dir), exist_ok=True)
                    torch.save({
                        'step': step,
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, os.path.join(str(SCRIPT_DIR), args.output_dir, args.saved_model))
                    print(f'Saved best model at step {step} | loss={val_loss:.4f}')

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1} | Train Loss: {avg_loss:.4f}')

def main():
    train(args)

if __name__ == "__main__":
    main()