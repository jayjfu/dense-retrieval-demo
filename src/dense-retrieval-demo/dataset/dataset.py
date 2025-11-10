from torch.utils.data import IterableDataset, Dataset
import json
import csv


class TokenizedTextPairDataset(IterableDataset):
    def __init__(self, data_path, limit=None):
        self.data_path = data_path
        self.limit = limit

    def __iter__(self):
        with open(self.data_path, 'r') as f:
            for i, line in enumerate(f):
                if self.limit and i >= self.limit:
                    break

                item = json.loads(line)
                query, pos, neg = item['query'], item['positive'], item['negative']

                yield {"input_ids": query + pos, "label": 1}
                yield {"input_ids": query + neg, "label": 0}

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