import torch


class HFBiEncoderCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        q_ids = [f["q_input_ids"] for f in features]
        p_ids = [f["p_input_ids"] for f in features]
        labels = [f["label"] for f in features]

        # Pad
        q_batch = self.tokenizer.pad({"input_ids": q_ids}, padding=True, return_tensors="pt")
        p_batch = self.tokenizer.pad({"input_ids": p_ids}, padding=True, return_tensors="pt")

        return {
            "q_input_ids": q_batch["input_ids"],
            "q_attention_mask": q_batch["attention_mask"],
            "p_input_ids": p_batch["input_ids"],
            "p_attention_mask": p_batch["attention_mask"],
            "label": torch.tensor(labels, dtype=torch.float),
        }