from torch.utils.data import Dataset
import json
import csv


class TokenizedTextPairDataset(Dataset):
    def __init__(self, data_path, limit=None):
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                query, pos, neg = item['query'], item['positive'], item['negative']

                self.data.append({"input_ids": query + pos, "label": 1})
                self.data.append({"input_ids": query + neg, "label": 0})

                if limit and len(self.data) >= 2 * limit:
                    break

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