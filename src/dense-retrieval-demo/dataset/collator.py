import torch

def pad_to_max_length(batch, pad_token_id=0):
    if 'input_ids' in batch[0]:  # cross-encoder
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['label'] for item in batch]

        max_len = max(len(seq) for seq in input_ids)
        padded_input_ids = [seq + [pad_token_id] * (max_len - len(seq)) for seq in input_ids]

        input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float)

        return {"input_ids": input_ids, "labels": labels}
    else: # bi-encoder
        query_input_ids = [item['query_input_ids'] for item in batch]
        passage_input_ids = [item['passage_input_ids'] for item in batch]
        labels = [item['label'] for item in batch]

        max_q = max(len(seq) for seq in query_input_ids)
        max_p = max(len(seq) for seq in passage_input_ids)
        max_len = max(max_q, max_p)

        q_padded = [seq + [pad_token_id] * (max_len - len(seq)) for seq in query_input_ids]
        p_padded = [seq + [pad_token_id] * (max_len - len(seq)) for seq in passage_input_ids]

        return {
            "query_input_ids": torch.tensor(q_padded, dtype=torch.long),
            "passage_input_ids": torch.tensor(p_padded, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
        }
