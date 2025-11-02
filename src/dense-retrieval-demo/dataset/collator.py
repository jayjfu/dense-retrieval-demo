import torch

def pad_to_max_length(batch, pad_token_id=0):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['label'] for item in batch]

    max_len = max(len(seq) for seq in input_ids)
    padded_input_ids = [seq + [pad_token_id] * (max_len - len(seq)) for seq in input_ids]

    input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return {"input_ids": input_ids, "labels": labels}