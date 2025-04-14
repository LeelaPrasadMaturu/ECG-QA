from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
from configs import config

class ClinicalTextProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.CLINICAL_BERT_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(config.CLINICAL_BERT_NAME)
        self.projection = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.Dropout(0.1))
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True)
        return self.projection(outputs.last_hidden_state[:, 0, :])