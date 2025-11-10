import argparse
import os
from transformers import AutoTokenizer
import json
from tqdm import tqdm
from multiprocessing import Pool, set_start_method, current_process


parser = argparse.ArgumentParser(description="prepare jsonl data for msmacro passage ranking")
parser.add_argument('--hf_tokenizer_name', default='bert-base-uncased', type=str)
parser.add_argument('--data_dir', default="./data", type=str)
parser.add_argument('--train_file', default="triples.train.small.tsv", type=str)
parser.add_argument('--max_length', default=128, type=int)
parser.add_argument('--output_path', default="./data/processed", type=str)
parser.add_argument('--output_file', default="ms_train.jsonl", type=str)
parser.add_argument('--num_processes', default=8, type=int)
parser.add_argument('--mp_chunk_size', default=100_000, type=int)
args = parser.parse_args()

def process_line(line):
    query, pos, neg = line.strip().split('\t')
    encoded = tokenizer([query, pos, neg], truncation=True, max_length=max_length)
    data = {
        "query": encoded['input_ids'][0],
        "positive": encoded['input_ids'][1],
        "negative": encoded['input_ids'][2]
    }

    with open(output_shard_file, 'a') as f_out:
        f_out.write(json.dumps(data) + '\n')

def init_worker(tokenizer_name, max_length_, output_dir_):
    global tokenizer, max_length, output_shard_file
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    max_length = max_length_
    pid = current_process().pid
    output_shard_file = os.path.join(output_dir_, f"ms_train_shard_{pid}.jsonl")

def merge_shards(output_dir, output_file):
    merged_file = os.path.join(output_dir, output_file)
    shard_files = [
        os.path.join(output_dir, f_name)
        for f_name in os.listdir(output_dir)
        if f_name.startswith("ms_train_shard_")
    ]

    with open(merged_file, 'w') as f_out:
        for shard_file in shard_files:
            with open(shard_file, 'r') as f_in:
                for line in f_in:
                    f_out.write(line)

    for shard_file in shard_files:
        os.remove(shard_file)

def prepare_json(args):
    train_data = os.path.join(args.data_dir, args.train_file)
    output_dir = os.path.join(args.output_path, "train")
    os.makedirs(output_dir, exist_ok=True)

    with open(train_data, 'rb') as f:
        total_lines = sum(1 for _ in f)

    with open(train_data, 'r') as f_in:
        with Pool(
                args.num_processes,
                initializer=init_worker,
                initargs=(args.hf_tokenizer_name, args.max_length, output_dir)
        ) as pool:
            for _ in tqdm(
                    pool.imap_unordered(process_line, f_in, chunksize=args.mp_chunk_size),
                    total=total_lines,
                    desc='Processing lines'
            ):
                pass

    # Merge
    merge_shards(output_dir, args.output_file)

def main():
    set_start_method('spawn', force=True)

    prepare_json(args)

if __name__ == "__main__":
    main()
