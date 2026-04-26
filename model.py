import torch
import torch.nn as nn
from constellations import get_constellation_size
from transformers import GPT2Config, GPT2Model


class ICLTransformer(nn.Module):
    def __init__(
        self,
        n_positions, 
        n_embd, 
        n_layer, 
        n_head, 
        n_classes
    ):

        super().__init__()

        config = GPT2Config(
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            vocab_size=1,
            use_cache=False,
        )

        self.n_classes = n_classes
        self.read_in = nn.Linear(self.n_classes, n_embd)
        self.backbone = GPT2Model(config)
        self.read_out = nn.Linear(n_embd, self.n_classes)

    def forward(self, y_batch, x_batch):
        z_batch = self._interleave(y_batch, x_batch)  # (B, 2T, num_classes)
        embeds = self.read_in(z_batch)                # (B, 2T, embed_dim)
        hidden = self.backbone(inputs_embeds=embeds).last_hidden_state
        logits = self.read_out(hidden)          # (B, 2T, num_classes)
        return logits[:, ::2, :]
    
    def _interleave(self, y_batch, x_batch):
        batch_size, num_points, num_classes = x_batch.size()
        _, _, y_classes = y_batch.size()

        if y_classes < num_classes:
            padding = torch.zeros(batch_size, num_points, num_classes - y_classes, device=y_batch.device,  dtype=y_batch.dtype)
            y_batch = torch.cat((y_batch, padding), dim=-1)
        
        interleaved = torch.stack((y_batch,x_batch), dim=2)
        interleaved = interleaved.view(batch_size, 2 * num_points, num_classes)
        return interleaved
    


def create_model(modulation_name, max_sequence_length=32):
    num_classes = get_constellation_size(modulation_name)

    model = ICLTransformer(
        n_positions=2 * max_sequence_length, 
        n_embd=64, 
        n_layer=8, 
        n_head=8, 
        n_classes=num_classes
    )

    return model
