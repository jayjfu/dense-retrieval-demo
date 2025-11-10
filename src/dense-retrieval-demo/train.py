import argparse
import os
import torch
from dataset import TokenizedTextPairDataset, pad_to_max_length
from models import BertForSequenceClassification, BertConfig, BertModel
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import glob
import re


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="train models for neural search task")
parser.add_argument('--data_path', default="../../benchmarks/msmarco-passage-ranking/data/processed/train/ms_train.jsonl", type=str)
parser.add_argument('--model_config', default="./data/pretrained_bert/config.json", type=str)
parser.add_argument('--model_weights', default="./data/pretrained_bert/pytorch_model.bin", type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--epochs', default=2, type=int)
parser.add_argument('--fine_tune_all', action='store_true')
parser.add_argument('--logging_steps', default=2_000, type=int)
parser.add_argument('--save_steps', default=20_000, type=int)
parser.add_argument('--resume_from_checkpoint', default=True, type=bool)
parser.add_argument('--output_dir', default="./checkpoints/custom-bert-msmarco", type=str)
args = parser.parse_args()

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "step": step,
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, scheduler, scaler, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler and checkpoint['scheduler']:
        scheduler.load_state_dict(checkpoint['scheduler'])
    if scaler and checkpoint['scaler']:
        scaler.load_state_dict(checkpoint['scaler'])
    start_epoch = checkpoint['epoch']
    global_step = checkpoint['step']
    print(f"Resumed from epoch {start_epoch}, step {global_step}")
    return start_epoch, global_step

def extract_step(path):
    m = re.search(r"ckpt_step(\d+)\.pt", path)
    return int(m.group(1)) if m else -1

def train(args):
    data_path = os.path.join(SCRIPT_DIR, args.data_path)
    dataset = TokenizedTextPairDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=pad_to_max_length)

    model_config = BertConfig(**json.load(open(os.path.join(SCRIPT_DIR, args.model_config))))
    model = BertForSequenceClassification(BertModel(model_config))

    model_weights = os.path.join(str(SCRIPT_DIR), args.model_weights)
    state_dict = torch.load(model_weights, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if not args.fine_tune_all:
        for param in model.bert.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    scaler = torch.amp.GradScaler('cuda')
    start_epoch = 0
    global_step = 0
    running_loss = 0

    if args.resume_from_checkpoint:
        ckpts = glob.glob(os.path.join(SCRIPT_DIR, args.output_dir, "ckpt_step*.pt"))
        if ckpts:
            latest = max(ckpts, key=extract_step)
            start_epoch, global_step = load_checkpoint(model, optimizer, scheduler, scaler, latest, device)

    with open(args.data_path, 'rb') as f:
        total_lines = sum(1 for _ in f)
    total_samples = total_lines * 2
    total_batches = (total_samples + args.batch_size - 1) // args.batch_size * args.epochs

    epoch_steps_done = global_step % (total_batches // args.epochs)
    global_step -= epoch_steps_done
    # resume_pbar = tqdm(total=epoch_steps_done, desc='Resuming')
    pbar = tqdm(total=total_batches, initial=global_step, desc='Training')

    for epoch in range(start_epoch, args.epochs):
        for batch in dataloader:
            global_step += 1

            if epoch_steps_done:
                epoch_steps_done -= 1
                # resume_pbar.update(1)
                pbar.update(1)
                continue

            input_ids, labels = batch['input_ids'].to(device), batch['labels'].to(device)

            model.train()
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, labels)
                loss = outputs['loss']

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if global_step % args.logging_steps == 0:
                logged_loss = running_loss / args.logging_steps
                print(f"Epoch [{epoch + 1}/{args.epochs}], Step [{global_step}], Logged Loss: {logged_loss:.4f}")
                running_loss = 0

            if global_step % args.save_steps == 0:
                os.makedirs(os.path.join(SCRIPT_DIR, args.output_dir), exist_ok=True)
                path = os.path.join(str(SCRIPT_DIR), args.output_dir, f"ckpt_step{global_step}.pt")
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, path)

            pbar.update(1)

def main():
    train(args)

if __name__ == "__main__":
    main()