import re
import unicodedata


def load_vocab(vocab_file):
    vocab = {}
    with open(vocab_file, 'r') as f:
        for i, token in enumerate(f.read().splitlines()):
            vocab[token] = i
    return vocab

def normalize(text):
    text = unicodedata.normalize('NFD', text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")  # remove accents
    return text.lower()

class BertTokenizer:
    def __init__(self, vocab_file):
        self.unk_token = '[UNK]'
        self.sep_token = '[SEP]'
        self.pad_token = '[PAD]'
        self.cls_token = '[CLS]'
        self.max_len = 512

        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize_wordpiece(self, word):
        if word in self.vocab:
            return [word]

        tokens = []
        start = 0

        while start < len(word):
            end = len(word)
            sub = None

            while start < end:
                piece = word[start:end]
                if start > 0:
                    piece = "##" + piece
                if piece in self.vocab:
                    sub = piece
                    break
                end -= 1

            if sub is None:
                return [self.unk_token]
            tokens.append(sub)
            start = end

        return tokens

    def tokenize(self, text):
        text = normalize(text)
        words = re.findall(r"\w+|[^\w\s]", text)  # basic token split (by all_special_tokens)
        tokens = []
        for word in words:
            tokens.extend(self.tokenize_wordpiece(word))
        return tokens

    def encode(self, text, max_length=128):
        tokens = [self.cls_token]
        wordpiece_tokens = self.tokenize(text)
        tokens.extend(wordpiece_tokens[: max_length - 2])
        tokens.append(self.sep_token)

        input_ids = [self.vocab.get(t, self.vocab[self.unk_token]) for t in tokens]
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids.extend([self.vocab[self.pad_token]] * pad_len)
            attention_mask.extend([0] * pad_len)
            token_type_ids.extend([0] * pad_len)

        return input_ids, attention_mask, token_type_ids
