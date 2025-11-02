import argparse
import os
from transformers import AutoTokenizer
import json
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


parser = argparse.ArgumentParser(description='prepare jsonl data for msmacro passage ranking')
parser.add_argument('--hf_tokenizer_name', default='bert-base-uncased', type=str)
parser.add_argument('--data_dir', default='data', type=str)
parser.add_argument('--train_file', default='triples.train.small.tsv', type=str)
parser.add_argument('--max_length', default=128, type=int)
parser.add_argument('--output_file', default='ms_train.jsonl', type=str)
parser.add_argument('--num_processes', default=8, type=int)
parser.add_argument('--mp_chunk_size', default=1024, type=int)
args = parser.parse_args()

def process_line(line, tokenizer, max_length):
    query, pos, neg = line.strip().split("\t")
    encoded = tokenizer([query, pos, neg], truncation=True, max_length=max_length)
    data = {'query': encoded['input_ids'][0], 'positive': encoded['input_ids'][1], 'negative': encoded['input_ids'][2]}

    return json.dumps(data)

def prepare_json(args):
    train_data = os.path.join(args.data_dir, args.train_file)
    output_dir = os.path.join(args.data_dir, 'processed', 'train')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.output_file)

    with open(train_data, 'rb') as f:
        total_lines = sum(1 for _ in f)

    with open(train_data, 'r') as f_in, open(output_path, 'w') as f_out:
        with Pool(args.num_processes) as pool:
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            worker = partial(process_line, tokenizer=tokenizer, max_length=args.max_length)

            for result in tqdm(pool.imap(worker, f_in, chunksize=args.mp_chunk_size), total=total_lines, desc='Processing lines'):
                f_out.write(result + '\n')

def main():
    prepare_json(args)

if __name__ == "__main__":
    main()
