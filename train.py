import torch
import torch.nn as nn
from dataset import build_dataset
from dataloader import create_dataloader
from model import create_model
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

def train_model(model, dataloader, num_epochs=15, lr=1e-4, device="cpu"):

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        total_samples = 0

        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_X) 
            final_logits = logits[:, -1, :] 
            loss = criterion(final_logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)
            total_samples += batch_X.size(0)

        avg_loss = total_loss / total_samples
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


def evaluate_model(model, dataloader, device="cpu"):

    model.to(device)
    model.eval()

    total_errors = 0
    total_samples = 0

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_X)

            final_logits = logits[:, -1, :]
            preds = torch.argmax(final_logits, dim=1)

            total_errors += (preds != batch_y).sum().item()
            total_samples += batch_y.size(0)

    ser = total_errors / total_samples
    return ser

def run_training():

    X, y = build_dataset(
        num_blocks=1000,
        block_length=30,
        modulation_name="BPSK",
        snr_db=10,
        num_pilots=3
    )

    loader = create_dataloader(X, y, batch_size=32)
    model = create_model("BPSK")
    train_model(model, loader, num_epochs=15)

    ser = evaluate_model(model, loader)
    print(f"Final SER: {ser:.4f}")


if __name__ == "__main__":
    run_training()