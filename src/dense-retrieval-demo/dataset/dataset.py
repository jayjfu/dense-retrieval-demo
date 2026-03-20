from torch.utils.data import IterableDataset, Dataset
import json
import csv


class TokenizedTextPairDataset(IterableDataset):
    def __init__(self, data_path, model_type="cross-encoder", limit=None):
        self.data_path = data_path
        self.model_type = model_type
        self.limit = limit

    def __iter__(self):
        with open(self.data_path, 'r') as f:
            for i, line in enumerate(f):
                if self.limit and i >= self.limit:
                    break

                item = json.loads(line)
                query_ids, pos_ids, neg_ids = item['query'], item['positive'], item['negative']

                if self.model_type == "cross-encoder":
                    yield {"input_ids": query_ids + pos_ids[1:], "label": 1}  # [CLS] query_tokens [SEP] passage_tokens [SEP]
                    yield {"input_ids": query_ids + neg_ids[1:], "label": 0}
                else:
                    yield {"query_input_ids": query_ids, "passage_input_ids": pos_ids, "label": 1}
                    yield {"query_input_ids": query_ids, "passage_input_ids": neg_ids, "label": 0}

class PassageDataset(Dataset):
    def __init__(self, data_path, limit=None):
        self.data = []
        with open(data_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                pid, passage = line[0], line[1]

                self.data.append(passage)

                if limit and len(self.data) >= limit:
                    break