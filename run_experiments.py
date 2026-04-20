import numpy as np
import matplotlib.pyplot as plt

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

def plot_ser_vs_snr(snr_list, ser_list, modulation, num_pilots):
    plt.figure()
    plt.plot(snr_list, ser_list)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.xticks(snr_list)
    plt.ylabel('Symbol Error Rate ($SER$)', fontsize=12)
    plt.yticks([0, 0.05, 0.1, 0.15, 0.2])
    plt.title(f'SER vs. SNR ({modulation}, k = {num_pilots})', fontsize=14)
    plt.grid()


def experiment_snr_sweep():
    print("=== SER vs SNR (MMSE) ===")

    modulation = "BPSK"
    snr_list = [0, 5, 10, 15, 20]
    ser_list = []
    
    num_pilots = 5

    for snr_db in snr_list:
        ser = run_mmse_experiment(
            modulation_name=modulation,
            snr_db=snr_db,
            num_blocks=10000,
            block_length=30,
            num_pilots=num_pilots
        )

        ser_list.append(ser)
        print(f"SNR={snr_db} dB, SER={ser:.5f}")
    
    plot_ser_vs_snr(snr_list, ser_list, modulation, num_pilots)

def plot_ser_vs_pilots(pilot_list, ser_list, modulation, snr_db):
    plt.figure()
    plt.plot(pilot_list, ser_list)
    plt.xlabel('Number of Pilots ($k$)', fontsize=12)
    plt.xticks(pilot_list)
    plt.ylabel('Symbol Error Rate ($SER$)', fontsize=12)
    plt.yticks([0, 0.05, 0.1, 0.15])
    plt.title(f'SER vs. Number of Pilots ({modulation}, SNR = {snr_db} dB)', fontsize=14)
    plt.grid()


def experiment_pilot_sweep():
    print("\n=== SER vs Number of Pilots (MMSE) ===")

    modulation = "BPSK"
    snr_db = 5
    pilot_list = [1, 2, 3, 4, 5]
    ser_list = []

    for k in pilot_list:
        ser = run_mmse_experiment(
            modulation_name=modulation,
            snr_db=snr_db,
            num_blocks=10000,
            block_length=30,
            num_pilots=k
        )

        ser_list.append(ser)
        print(f"Pilots={k}, SER={ser:.5f}")
    
    plot_ser_vs_pilots(pilot_list, ser_list, modulation, snr_db)

def plot_ser_vs_modulation(modulations, ser_list, snr_db, num_pilots):
    plt.figure()
    plt.bar(modulations, ser_list)
    plt.xlabel('Modulation Scheme', fontsize=12)
    plt.xticks(modulations)
    plt.ylabel('Symbol Error Rate ($SER$)', fontsize=12)
    plt.yticks([0, 0.1, 0.2, 0.3])
    plt.title(f'SER vs. Modulation (k = {num_pilots}, SNR = {snr_db} dB)', fontsize=14)
    plt.grid()


def experiment_modulation_comparison():
    print("\n=== SER vs Modulation ===")

    snr_db = 20
    num_pilots = 5
    modulations = ["BPSK", "QPSK", "16QAM", "64QAM"]
    ser_list = []

    for mod in modulations:
        ser = run_mmse_experiment(
            modulation_name=mod,
            snr_db=snr_db,
            num_blocks=10000,
            block_length=30,
            num_pilots=num_pilots
        )

        ser_list.append(ser)
        print(f"{mod}, SER={ser:.5f}")
    
    plot_ser_vs_modulation(modulations, ser_list, snr_db, num_pilots)


if __name__ == "__main__":
    experiment_snr_sweep()
    experiment_pilot_sweep()
    experiment_modulation_comparison()
    plt.show()