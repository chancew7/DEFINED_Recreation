import torch
import torch.nn as nn
from constellations import get_constellation_size
import math


class ICLTransformer(nn.Module):
    def __init__(
        self,
        input_dim=2,
        embed_dim=64,
        num_heads=8,
        num_layers=4,
        num_classes=2
    ):

        super().__init__()

        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        
        seq_len = x.size(1)

        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        logits = self.classifier(x)   

        return logits
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
    

def create_model(modulation_name):
    num_classes = get_constellation_size(modulation_name)

    model = ICLTransformer(
        input_dim=3,
        embed_dim=64,
        num_heads=8,
        num_layers=4,
        num_classes=num_classes
    )

    return model
