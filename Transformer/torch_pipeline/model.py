import torch.nn as nn
from transformers import BertModel

class SentimentModel(nn.Module):
    def __init__(self, model_name: str, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return self.classifier(self.dropout(pooled_output))
