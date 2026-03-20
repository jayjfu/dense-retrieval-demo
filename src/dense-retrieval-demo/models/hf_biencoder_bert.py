import torch
import torch.nn.functional as F
from transformers import AutoModel


class HFBiEncoderModel(torch.nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.config = self.base_model.config  # needed for Trainer compatibility

    def forward(self, q_input_ids, p_input_ids, label=None, q_attention_mask=None, p_attention_mask=None):
        q_out = self.base_model(input_ids=q_input_ids, attention_mask=q_attention_mask)
        p_out = self.base_model(input_ids=p_input_ids, attention_mask=p_attention_mask)

        # Use [CLS] token embeddings
        q_emb = q_out.last_hidden_state[:, 0]
        p_emb = p_out.last_hidden_state[:, 0]

        sim_logits = F.cosine_similarity(q_emb, p_emb, dim=1)

        loss = None
        if label is not None:
            loss = F.binary_cross_entropy_with_logits(sim_logits, label.float(), reduction='mean')

        sim_probe = torch.sigmoid(sim_logits)

        return {"loss": loss, "logits": sim_probe} if loss is not None else {"logits": sim_probe}

    @classmethod
    def from_pretrained(cls, model_name):
        return cls(model_name)