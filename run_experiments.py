import numpy as np

from channel import generate_block
from constellations import get_constellation
from mmse import mmse_estimate, detect_symbol


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


def experiment_snr_sweep():
    print("=== SER vs SNR (MMSE) ===")

    modulation = "BPSK"
    snr_list = [0, 5, 10, 15, 20]
    num_pilots = 3

    for snr_db in snr_list:
        ser = run_mmse_experiment(
            modulation_name=modulation,
            snr_db=snr_db,
            num_blocks=2000,
            block_length=30,
            num_pilots=num_pilots
        )

        print(f"SNR={snr_db} dB, SER={ser:.5f}")


def experiment_pilot_sweep():
    print("\n=== SER vs Number of Pilots (MMSE) ===")

    modulation = "BPSK"
    snr_db = 10
    pilot_list = [1, 2, 3, 4, 5]

    for k in pilot_list:
        ser = run_mmse_experiment(
            modulation_name=modulation,
            snr_db=snr_db,
            num_blocks=2000,
            block_length=30,
            num_pilots=k
        )

        print(f"Pilots={k}, SER={ser:.5f}")


def experiment_modulation_comparison():
    print("\n=== SER vs Modulation ===")

    snr_db = 10
    num_pilots = 3
    modulations = ["BPSK", "QPSK", "16QAM", "64QAM"]

    for mod in modulations:
        ser = run_mmse_experiment(
            modulation_name=mod,
            snr_db=snr_db,
            num_blocks=2000,
            block_length=30,
            num_pilots=num_pilots
        )

        print(f"{mod}, SER={ser:.5f}")


if __name__ == "__main__":
    experiment_snr_sweep()
    experiment_pilot_sweep()
    experiment_modulation_comparison()