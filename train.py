import torch
import torch.nn as nn
from dataset import build_dataset
from dataloader import create_dataloader
from model import create_model
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

def icl_train_step(model, y_batch, x_batch, labels, optimizer, criterion, num_pilots):
    model.train()

    logits = model(y_batch, x_batch)  # clean ground-truth x context

    B, T, C = logits.shape

    payload_logits = logits[:, num_pilots:, :].reshape(-1, C)
    payload_labels = labels[:, num_pilots:].reshape(-1)

    loss = criterion(payload_logits, payload_labels).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def defined_train_step(model, y_batch, x_batch, labels, optimizer, criterion,
                       num_pilots, loss_weight=0.7):
    model.train()

    B, T, C = x_batch.shape
    x_feedback = torch.zeros_like(x_batch)

    with torch.no_grad():
        for t in range(T):
            if t < num_pilots:
                x_feedback[:, t, :] = x_batch[:, t, :]
            else:
                logits = model(y_batch, x_feedback)
                pred_labels = torch.argmax(logits[:, t, :], dim=-1)
                x_feedback[:, t, :] = torch.nn.functional.one_hot(
                    pred_labels, num_classes=C
                ).float()

    logits_df = model(y_batch, x_feedback)
    loss_df = criterion(
        logits_df[:, num_pilots:, :].reshape(-1, C),
        labels[:, num_pilots:].reshape(-1)
    ).mean()

    logits_clean = model(y_batch, x_batch)
    loss_clean = criterion(
        logits_clean[:, num_pilots:, :].reshape(-1, C),
        labels[:, num_pilots:].reshape(-1)
    ).mean()

    loss = loss_weight * loss_df + (1.0 - loss_weight) * loss_clean

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def icl_evaluate(model, dataloader, num_pilots, device="cpu"):
    model.eval()
    total_errors = 0
    total_symbols = 0

    with torch.no_grad():
        for y_batch, x_batch, labels_batch in dataloader:
            y_batch = y_batch.to(device)
            x_batch = x_batch.to(device)
            labels_batch = labels_batch.to(device)

            B, T, C = x_batch.shape

            for t in range(num_pilots, T):
                y_prompt = torch.zeros_like(y_batch)
                x_prompt = torch.zeros_like(x_batch)

                y_prompt[:, :num_pilots, :] = y_batch[:, :num_pilots, :]
                x_prompt[:, :num_pilots, :] = x_batch[:, :num_pilots, :]

                y_prompt[:, t, :] = y_batch[:, t, :]

                logits = model(y_prompt, x_prompt)
                preds = torch.argmax(logits[:, t, :], dim=-1)

                total_errors += (preds != labels_batch[:, t]).sum().item()
                total_symbols += B

    return total_errors / total_symbols


def defined_evaluate(model, dataloader, num_pilots, device="cpu"):
    model.eval()
    total_errors = 0
    total_symbols = 0

    with torch.no_grad():
        for y_batch, x_batch, labels_batch in dataloader:
            y_batch = y_batch.to(device)
            x_batch = x_batch.to(device)
            labels_batch = labels_batch.to(device)

            B, T, C = x_batch.shape
            x_feedback = torch.zeros_like(x_batch)

            for t in range(T):
                if t < num_pilots:
                    x_feedback[:, t, :] = x_batch[:, t, :]
                else:
                    logits = model(y_batch, x_feedback)
                    pred_labels = torch.argmax(logits[:, t, :], dim=-1)

                    x_feedback[:, t, :] = torch.nn.functional.one_hot(
                        pred_labels, num_classes=C
                    ).float()

                    total_errors += (pred_labels != labels_batch[:, t]).sum().item()
                    total_symbols += B

    return total_errors / total_symbols



def train_icl_model(num_blocks=1000, block_length=31,
                    modulation_name="BPSK", num_pilots=1,
                    snr_db_min=0, snr_db_max=20,
                    batch_size=128,
                    num_epochs=1000,
                    learning_rate=1e-4,
                    log_every=100,
                    n_it_per_epoch=50,
                    early_stopping_patience=5,
                    min_ser_improvement=1e-4,
                    device="cpu"):

    print(f"=== Training Vanilla ICL | {modulation_name} | pilots={num_pilots} ===")

    y_seqs, x_seqs, labels = build_dataset(
        num_blocks=num_blocks,
        block_length=block_length,
        modulation_name=modulation_name,
        snr_db_min=snr_db_min,
        snr_db_max=snr_db_max,
        num_pilots=num_pilots,
        seed=42,
    )

    split_idx = int(0.8 * len(labels))

    train_loader = create_dataloader(
        y_seqs[:split_idx],
        x_seqs[:split_idx],
        labels[:split_idx],
        batch_size=batch_size,
    )

    val_loader = create_dataloader(
        y_seqs[split_idx:],
        x_seqs[split_idx:],
        labels[split_idx:],
        batch_size=len(y_seqs[split_idx:]),
    )

    model = create_model(modulation_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(reduction="none")

    best_val_ser = 1.0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (y_batch, x_batch, batch_labels) in enumerate(train_loader):
            if batch_idx >= n_it_per_epoch:
                break

            y_batch = y_batch.to(device)
            x_batch = x_batch.to(device)
            batch_labels = batch_labels.to(device)

            loss = icl_train_step(
                model,
                y_batch,
                x_batch,
                batch_labels,
                optimizer,
                criterion,
                num_pilots,
            )


            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        if epoch % log_every == 0 or epoch == num_epochs - 1:
            val_ser = icl_evaluate(
                model,
                val_loader,
                num_pilots,
                device,
            )

            print(
                f"Epoch {epoch:4d} [ICL] | "
                f"Loss: {avg_loss:.4f} | "
                f"Val SER: {val_ser:.4f}"
            )

            if val_ser < best_val_ser - min_ser_improvement:
                best_val_ser = val_ser
                epochs_without_improvement = 0

                model_path = f"best_icl_{modulation_name}_k{num_pilots}.pth"
                torch.save(model.state_dict(), model_path)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"Early stopping ICL at epoch {epoch}. "
                    f"Best val SER: {best_val_ser:.4f}"
                )
                break

    print(f"\n*** ICL training complete. Best val SER: {best_val_ser:.4f} ***")
    return model



def train_defined_model(num_blocks=1000, block_length=31,
                        modulation_name="BPSK", num_pilots=1,
                        snr_db_min=0, snr_db_max=20,
                        batch_size=128,
                        num_epochs=1000,
                        icl_warmup_epochs=100,
                        loss_weight=0.7,
                        learning_rate=1e-4,
                        log_every=100,
                        n_it_per_epoch=50,
                        early_stopping_patience=5,
                        min_ser_improvement=1e-4,
                        device="cpu"):

    print(f"=== Training DEFINED | {modulation_name} | pilots={num_pilots} ===")

    y_seqs, x_seqs, labels = build_dataset(
        num_blocks=num_blocks,
        block_length=block_length,
        modulation_name=modulation_name,
        snr_db_min=snr_db_min,
        snr_db_max=snr_db_max,
        num_pilots=num_pilots,
        seed=42,
    )

    split_idx = int(0.8 * len(labels))

    train_loader = create_dataloader(
        y_seqs[:split_idx],
        x_seqs[:split_idx],
        labels[:split_idx],
        batch_size=batch_size,
    )

    val_loader = create_dataloader(
        y_seqs[split_idx:],
        x_seqs[split_idx:],
        labels[split_idx:],
        batch_size=len(y_seqs[split_idx:]),
    )

    model = create_model(modulation_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(reduction="none")

    best_val_ser = 1.0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        if epoch < icl_warmup_epochs:
            mode = "ICL warmup"
        else:
            mode = "DEFINED"

            if epoch == icl_warmup_epochs:
                print(f"\n*** Switching to DEFINED training at epoch {epoch} ***\n")
                epochs_without_improvement = 0

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (y_batch, x_batch, batch_labels) in enumerate(train_loader):
            if batch_idx >= n_it_per_epoch:
                break

            y_batch = y_batch.to(device)
            x_batch = x_batch.to(device)
            batch_labels = batch_labels.to(device)

            if epoch < icl_warmup_epochs:
                loss = icl_train_step(
                    model,
                    y_batch,
                    x_batch,
                    batch_labels,
                    optimizer,
                    criterion,
                    num_pilots,
                )
            else:
                loss = defined_train_step(
                    model,
                    y_batch,
                    x_batch,
                    batch_labels,
                    optimizer,
                    criterion,
                    num_pilots,
                    loss_weight,
                )

            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        if epoch % log_every == 0 or epoch == num_epochs - 1:
            val_ser = defined_evaluate(
                model,
                val_loader,
                num_pilots,
                device,
            )

            print(
                f"Epoch {epoch:4d} [{mode:10s}] | "
                f"Loss: {avg_loss:.4f} | "
                f"Val SER: {val_ser:.4f}"
            )

            if epoch >= icl_warmup_epochs:
                if val_ser < best_val_ser - min_ser_improvement:
                    best_val_ser = val_ser
                    epochs_without_improvement = 0

                    model_path = f"best_defined_{modulation_name}_k{num_pilots}.pth"
                    torch.save(model.state_dict(), model_path)
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= early_stopping_patience:
                    print(
                        f"Early stopping DEFINED at epoch {epoch}. "
                        f"Best val SER: {best_val_ser:.4f}"
                    )
                    break

    print(f"\n*** DEFINED training complete. Best val SER: {best_val_ser:.4f} ***")
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    experiment_settings = [
    {
        "modulation_name": "QPSK",
        "num_pilots": 1,
        "num_blocks": 2000,
        "batch_size": 512,
        "num_epochs": 800,
        "icl_warmup_epochs": 150,
        "n_it_per_epoch": 30,
        "icl_log_every": 100,
        "defined_log_every": 100,
        "early_stopping_patience": 5,
    },
    {
        "modulation_name": "QPSK",
        "num_pilots": 2,
        "num_blocks": 2000,
        "batch_size": 512,
        "num_epochs": 800,
        "icl_warmup_epochs": 150,
        "n_it_per_epoch": 30,
        "icl_log_every": 100,
        "defined_log_every": 100,
        "early_stopping_patience": 5,
    },
    {
        "modulation_name": "QPSK",
        "num_pilots": 3,
        "num_blocks": 2000,
        "batch_size": 512,
        "num_epochs": 700,
        "icl_warmup_epochs": 150,
        "n_it_per_epoch": 30,
        "icl_log_every": 100,
        "defined_log_every": 100,
        "early_stopping_patience": 5,
    },
    {
        "modulation_name": "QPSK",
        "num_pilots": 4,
        "num_blocks": 2000,
        "batch_size": 512,
        "num_epochs": 700,
        "icl_warmup_epochs": 150,
        "n_it_per_epoch": 30,
        "icl_log_every": 100,
        "defined_log_every": 100,
        "early_stopping_patience": 5,
    },
    {
        "modulation_name": "QPSK",
        "num_pilots": 5,
        "num_blocks": 2000,
        "batch_size": 512,
        "num_epochs": 700,
        "icl_warmup_epochs": 150,
        "n_it_per_epoch": 30,
        "icl_log_every": 100,
        "defined_log_every": 100,
        "early_stopping_patience": 5,
    },
    {
        "modulation_name": "BPSK",
        "num_pilots": 2,
        "num_blocks": 2000,
        "batch_size": 512,
        "num_epochs": 700,
        "icl_warmup_epochs": 150,
        "n_it_per_epoch": 30,
        "icl_log_every": 100,
        "defined_log_every": 100,
        "early_stopping_patience": 5,
    },
    {
        "modulation_name": "16QAM",
        "num_pilots": 2,
        "num_blocks": 4000,
        "batch_size": 512,
        "num_epochs": 1200,
        "icl_warmup_epochs": 250,
        "n_it_per_epoch": 40,
        "icl_log_every": 150,
        "defined_log_every": 150,
        "early_stopping_patience": 4,
    },
    {
        "modulation_name": "64QAM",
        "num_pilots": 2,
        "num_blocks": 6000,
        "batch_size": 512,
        "num_epochs": 1600,
        "icl_warmup_epochs": 300,
        "n_it_per_epoch": 50,
        "icl_log_every": 200,
        "defined_log_every": 200,
        "early_stopping_patience": 3,
    },
]


    for setting in experiment_settings:
        modulation_name = setting["modulation_name"]
        num_pilots = setting["num_pilots"]

        print(
            f"\n\n=== Starting setting: {modulation_name}, "
            f"k={num_pilots} ==="
        )

        train_icl_model(
            num_blocks=setting["num_blocks"],
            modulation_name=setting["modulation_name"],
            num_pilots=setting["num_pilots"],
            block_length=31,
            snr_db_min=0,
            snr_db_max=20,
            batch_size=setting["batch_size"],
            num_epochs=setting["num_epochs"],
            learning_rate=1e-4,
            log_every=setting["icl_log_every"],
            n_it_per_epoch=setting["n_it_per_epoch"],
            early_stopping_patience=setting["early_stopping_patience"],
            device=device,
        )


        train_defined_model(
            num_blocks=setting["num_blocks"],
            modulation_name=setting["modulation_name"],
            num_pilots=setting["num_pilots"],
            block_length=31,
            snr_db_min=0,
            snr_db_max=20,
            batch_size=setting["batch_size"],
            num_epochs=setting["num_epochs"],
            icl_warmup_epochs=setting["icl_warmup_epochs"],
            loss_weight=0.7,
            learning_rate=1e-4,
            log_every=setting["defined_log_every"],
            n_it_per_epoch=setting["n_it_per_epoch"],
            early_stopping_patience=setting["early_stopping_patience"],
            device=device,
        )
