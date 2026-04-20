import torch
import torch.nn as nn
from constellations import get_constellation_size


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


        x = self.input_proj(x)  # (batch, seq_len, embed_dim)
        x = self.transformer(x)
        x = x[:, -1, :]

        logits = self.classifier(x)
        return logits
    
def create_model(modulation_name):
    num_classes = get_constellation_size(modulation_name)

    model = ICLTransformer(
        input_dim=2,
        embed_dim=64,
        num_heads=8,
        num_layers=4,
        num_classes=num_classes
    )

    return model
