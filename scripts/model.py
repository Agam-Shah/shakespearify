import torch
import torch.nn as nn
from transformers import BertModel

class BERT2Transformer(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", hidden_dim=768, num_layers=2, num_heads=8):
        super(BERT2Transformer, self).__init__()

        # Load Pretrained BERT (Encoder)
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Transformer Decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        # Final Linear Layer to Predict Tokens
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, modern_embedding, shakespeare_embedding):

        # Pass modern English input through BERT
        encoder_output = self.bert(inputs_embeds=modern_embedding.unsqueeze(1)).last_hidden_state

        # Transformer decoder
        decoder_output = self.transformer_decoder(shakespeare_embedding.unsqueeze(1), encoder_output)

        # Final Linear layer
        output = self.fc_out(decoder_output)

        return output



