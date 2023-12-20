import torch
import torch.nn as nn
from transformers import BertConfig
from transformers import BertForTokenClassification


class FacEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, u_weight, v_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.u = nn.Embedding(vocab_size, hidden_size)
        self.v = nn.Linear(hidden_size, emb_size)
        self.u.weight.data = u_weight
        self.v.weight.data = v_weight.T
        
        self.hidden_size = hidden_size
    
    def forward(self, x):
        return self.v(self.u(x))

class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id2label = {0: 'B-LOC',
                  1: 'B-MISC',
                  2: 'B-ORG',
                  3: 'B-PER',
                  4: 'I-LOC',
                  5: 'I-MISC',
                  6: 'I-ORG',
                  7: 'I-PER',
                  8: 'O'}
        self.label2id = {'B-LOC': 0,
                  'B-MISC': 1,
                  'B-ORG': 2,
                  'B-PER': 3,
                  'I-LOC': 4,
                  'I-MISC': 5,
                  'I-ORG': 6,
                  'I-PER': 7,
                  'O': 8}
        
        conf = BertConfig(path='bert-base-cased', architectures="BertForMaskedLM", id2label=self.id2label, label2id=self.label2id, vocab_size=28996, gradient_checkpointing=False)
        self.bert = BertForTokenClassification(conf)
        self.bert.bert.embeddings.word_embeddings = FacEmbedding(conf.vocab_size, conf.hidden_size, 128, u_weight=torch.randn(conf.vocab_size, 128), v_weight=torch.randn(128, conf.hidden_size))

        for i in range(1, len(self.bert.bert.encoder.layer)):
            self.bert.bert.encoder.layer[i].attention.self.query = self.bert.bert.encoder.layer[0].attention.self.query
            self.bert.bert.encoder.layer[i].attention.self.key = self.bert.bert.encoder.layer[0].attention.self.key
            self.bert.bert.encoder.layer[i].attention.self.value = self.bert.bert.encoder.layer[0].attention.self.value
            self.bert.bert.encoder.layer[i].intermediate = self.bert.bert.encoder.layer[0].intermediate
            self.bert.bert.encoder.layer[i].output.dense = self.bert.bert.encoder.layer[0].output.dense
    
    def forward(self, input_ids, attention_mask, **kwargs):
        return torch.nn.functional.softmax(self.bert(input_ids=input_ids, attention_mask=attention_mask).logits, dim=-1)
    
    def load_state_dict(self, x):
        return self.bert.load_state_dict(x)