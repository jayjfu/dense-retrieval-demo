import torch.nn as nn


# Bert classifier head
class BertForSequenceClassification(nn.Module):
    def __init__(self, bert, num_labels=2, dropout=0.1):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, labels, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_hidden_state = outputs[1]
        x = self.dropout(cls_hidden_state)
        logits = self.classifier(x)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        return {"logits": logits, "loss": loss}
