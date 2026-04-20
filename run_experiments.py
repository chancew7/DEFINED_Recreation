import numpy as np

from channel import generate_block
from constellations import get_constellation
from mmse import mmse_estimate, detect_symbol
from dataset import build_dataset
from dataloader import create_dataloader
from model import create_model
from train import train_model, evaluate_model


def compute_ser(num_errors, num_total):
    return num_errors / num_total


def run_mmse_experiment(
    modulation_name,
    snr_db,
    num_blocks,
    block_length,
    num_pilots
):
    rng = np.random.default_rng()

    constellation = get_constellation(modulation_name)

    total_errors = 0
    total_symbols = 0

    for _ in range(num_blocks):
        block = generate_block(
            block_length=block_length,
            modulation_name=modulation_name,
            snr_db=snr_db,
            r=rng
        )

        x = block["transmitted_symbols"]
        y = block["received_symbols"]
        labels = block["transmitted_labels"]
        noise_variance = block["noise_variance"]

        pilot_x = x[:num_pilots]
        pilot_y = y[:num_pilots]

        H_hat = mmse_estimate(pilot_x, pilot_y, noise_variance)

        for i in range(num_pilots, block_length):
            y_i = y[i]

            pred_label = detect_symbol(y_i, H_hat, constellation)
            true_label = labels[i]

            if pred_label != true_label:
                total_errors += 1

            total_symbols += 1

    return compute_ser(total_errors, total_symbols)


def run_transformer_experiment(
    modulation_name,
    snr_db,
    num_blocks,
    block_length,
    num_pilots
):

    X, y = build_dataset(
        num_blocks=num_blocks,
        block_length=block_length,
        modulation_name=modulation_name,
        snr_db=snr_db,
        num_pilots=num_pilots
    )

    loader = create_dataloader(X, y, batch_size=32)
    model = create_model(modulation_name)
    train_model(model, loader, num_epochs=15)

    ser = evaluate_model(model, loader)

    return ser


def experiment_snr_sweep():
    print("=== SER vs SNR (MMSE vs Transformer) ===")

    modulation = "BPSK"
    snr_list = [0, 5, 10, 15, 20]
    k = 3

    for snr_db in snr_list:
        mmse_ser = run_mmse_experiment(
            modulation, snr_db, 1000, 30, k
        )

        transformer_ser = run_transformer_experiment(
            modulation, snr_db, 1000, 30, k
        )

        print(f"SNR={snr_db} | MMSE={mmse_ser:.4f} | Transformer={transformer_ser:.4f}")


def experiment_pilot_sweep():
    print("\n=== SER vs Pilots (MMSE vs Transformer) ===")

    modulation = "BPSK"
    snr_db = 10
    pilot_list = [1, 2, 3, 4, 5]

    for k in pilot_list:
        mmse_ser = run_mmse_experiment(
            modulation, snr_db, 1000, 30, k
        )

        transformer_ser = run_transformer_experiment(
            modulation, snr_db, 1000, 30, k
        )

        print(f"k={k} | MMSE={mmse_ser:.4f} | Transformer={transformer_ser:.4f}")


def experiment_modulation():
    print("\n=== SER vs Modulation ===")

    snr_db = 10
    k = 3
    modulations = ["BPSK", "QPSK", "16QAM"]

    for mod in modulations:
        mmse_ser = run_mmse_experiment(
            mod, snr_db, 1000, 30, k
        )

        transformer_ser = run_transformer_experiment(
            mod, snr_db, 1000, 30, k
        )

        print(f"{mod} | MMSE={mmse_ser:.4f} | Transformer={transformer_ser:.4f}")


if __name__ == "__main__":
    experiment_snr_sweep()
    experiment_pilot_sweep()
    experiment_modulation()