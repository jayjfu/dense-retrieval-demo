import torch
import torch.nn.functional as F


class BiEncoderModel(torch.nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def forward(self, query_input_ids, passage_input_ids, labels=None, query_attention_mask=None, passage_attention_mask=None):
        q_out = self.bert(input_ids=query_input_ids)
        p_out = self.bert(input_ids=passage_input_ids)

        q_cls = q_out[0][:, 0]
        p_cls = p_out[0][:, 0]

        sim_logits = F.cosine_similarity(q_cls, p_cls, dim=1)

        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(sim_logits, labels.float(), reduction='mean')

        sim_probe = torch.sigmoid(sim_logits)

        return {"loss": loss, "logits": sim_probe} if loss is not None else {"logits": sim_probe}