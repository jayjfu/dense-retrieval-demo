import argparse
import os
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import datasets


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='train models for neural search task')
parser.add_argument('--data_path', default='../../benchmarks/msmarco-passage-ranking/data/processed/train/ms_train_10k.jsonl', type=str)
parser.add_argument('--lr', default=5e-6, type=float)
parser.add_argument('--epochs', default=2, type=int)
parser.add_argument('--output_dir', default='./checkpoints', type=str)
args = parser.parse_args()

def train(args):
    data_files = {'train': os.path.join(SCRIPT_DIR, args.data_path)}
    dataset = datasets.load_dataset('json', data_files=data_files)
    dataset = dataset['train']

    def gen_pairs(batch):
        input_ids, labels = [], []

        for q, p, n in zip(batch['query'], batch['positive'], batch['negative']):
            input_ids.append(q + p)
            labels.append(1)
            input_ids.append(q + n)
            labels.append(0)

        return {'input_ids': input_ids, 'label': labels}

    dataset = dataset.map(gen_pairs, batched=True, remove_columns=dataset.column_names)
    dataset = dataset.with_format(type='torch')

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    training_args = TrainingArguments(
        output_dir=os.path.join(str(SCRIPT_DIR), args.output_dir, 'bert-msmarco'),
        do_train=True,
        per_device_train_batch_size=8,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        save_steps=2000,
        fp16=True if torch.cuda.is_available() else False,
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator
    )
    trainer.train()

def main():
    train(args)

if __name__ == "__main__":
    main()