import argparse
import os
from dataset import HFBiEncoderCollator
from models import HFBiEncoderModel
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import datasets


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="train models for neural search task")
parser.add_argument('--data_path', default="../../benchmarks/msmarco-passage-ranking/data/processed/train/ms_train.jsonl", type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=5e-6, type=float)
parser.add_argument('--num_epochs', default=1, type=int)
parser.add_argument('--logging_steps', default=2_000, type=int)
parser.add_argument('--save_steps', default=20_000, type=int)
parser.add_argument('--model_type', default="cross-encoder", choices=["cross-encoder", "bi-encoder"])
parser.add_argument('--fine_tune_all', action='store_true')
parser.add_argument('--no_resume', action='store_true')
parser.add_argument('--output_dir', default="./checkpoints/bert-msmarco", type=str)
args = parser.parse_args()

def train(args):
    data_files = {"train": os.path.join(SCRIPT_DIR, args.data_path)}
    dataset = datasets.load_dataset('json', data_files=data_files)
    dataset = dataset['train']

    if args.model_type == "cross-encoder":
        def gen_cross_pairs(batch):
            input_ids, labels = [], []
            for q, p, n in zip(batch['query'], batch['positive'], batch['negative']):
                input_ids.append(q + p[1:])  # [CLS] query_tokens [SEP] passage_tokens [SEP]
                labels.append(1)
                input_ids.append(q + n[1:])
                labels.append(0)

            return {"input_ids": input_ids, "label": labels}

        dataset = dataset.map(gen_cross_pairs, batched=True, remove_columns=dataset.column_names)
    else:
        def gen_bi_pairs(batch):
            q_input_ids, p_input_ids, labels = [], [], []
            for q, p, n in zip(batch['query'], batch['positive'], batch['negative']):
                q_input_ids.append(q)
                p_input_ids.append(p)
                labels.append(1)
                q_input_ids.append(q)
                p_input_ids.append(n)
                labels.append(0)

            return {"q_input_ids": q_input_ids, "p_input_ids": p_input_ids, "label": labels}

        dataset = dataset.map(gen_bi_pairs, batched=True, remove_columns=dataset.column_names)

    dataset = dataset.with_format(type='torch')

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    if args.model_type=="cross-encoder":
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
    else:
        model = HFBiEncoderModel.from_pretrained("bert-base-uncased")
        collator = HFBiEncoderCollator(tokenizer=tokenizer)

    if not args.fine_tune_all:
        if args.model_type == "bi-encoder":
            raise ValueError("Bi-Encoder can only be used in --fine_tune_all mode")

        for param in model.base_model.parameters():
            param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=os.path.join(str(SCRIPT_DIR), args.output_dir),
        do_train=True,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        warmup_ratio=0.1,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=True,
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator
    )
    trainer.train(resume_from_checkpoint=not args.no_resume)

def main():
    train(args)

if __name__ == "__main__":
    main()