import torch
import torch.nn as nn
from bert_pretraining.bert_embedding import BertEmbedding
from transformer_layers import TransformerEncoderLayer

class PreTrainingBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = BertEmbedding(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])

        # for NSP task
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        self.nsp_output_layer = nn.Linear(config.hidden_size, 2)

        # for Masked LM task
        self.mlm_transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=1e-12),
        )
        self.mlm_output_layer = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mlm_output_layer.weight = self.embedding.token_embedding.weight
        self.mlm_bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, input_ids, segment_ids, attn_mask):
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2) # attn_mask.shape: (batch_size, 1, 1, seq_len)
        hidden_state = self.embedding(input_ids, segment_ids)

        for mod in self.layers:
            hidden_state = mod(hidden_state, attn_mask)

        output = hidden_state
        pooled_output = self.pooler(output[:, 0]) # [CLS] token hidden state

        nsp_logits = self.nsp_output_layer(pooled_output)
        mlm_logits = self.mlm_output_layer(self.mlm_transform(output)) + self.mlm_bias
        return mlm_logits, nsp_logits