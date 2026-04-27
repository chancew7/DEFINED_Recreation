from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from channel import generate_block
from constellations import get_constellation
from dataloader import create_dataloader
from dataset import build_dataset
from mmse import detect_symbol, evaluate_mmse, mmse_estimate
from model import create_model
from train import defined_evaluate, icl_evaluate


PROJECT_DIR = Path(__file__).resolve().parent
PLOT_DIR = PROJECT_DIR / "experiment_plots"

BLOCK_LENGTH = 31
TEST_BLOCKS = 1000
TEST_SEED = 123

PILOT_SWEEP_MODULATION = "QPSK"
MODULATION_SWEEP_PILOTS = 2
FIXED_SNR_DB = 10

PILOT_COUNTS = [1, 2, 3, 4, 5]
MODULATIONS = ["BPSK", "QPSK", "16QAM", "64QAM"]
SNR_VALUES = [0, 5, 10, 15, 20]


def make_test_loader(modulation_name, num_pilots, snr_db, batch_size=512):
    y_test, x_test, labels_test = build_dataset(
        num_blocks=TEST_BLOCKS,
        block_length=BLOCK_LENGTH,
        modulation_name=modulation_name,
        snr_db_min=snr_db,
        snr_db_max=snr_db,
        num_pilots=num_pilots,
        seed=TEST_SEED,
    )

    return create_dataloader(
        y_test,
        x_test,
        labels_test,
        batch_size=batch_size,
    )


def load_trained_model(method, modulation_name, num_pilots, device):
    model_path = PROJECT_DIR / f"best_{method}_{modulation_name}_k{num_pilots}.pth"

    if not model_path.exists():
        print(f"Missing checkpoint: {model_path.name}")
        return None

    model = create_model(modulation_name).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def evaluate_icl_checkpoint(modulation_name, num_pilots, snr_db, device):
    model = load_trained_model("icl", modulation_name, num_pilots, device)

    if model is None:
        return np.nan

    loader = make_test_loader(modulation_name, num_pilots, snr_db)
    return icl_evaluate(model, loader, num_pilots, device)


def evaluate_defined_checkpoint(modulation_name, num_pilots, snr_db, device):
    model = load_trained_model("defined", modulation_name, num_pilots, device)

    if model is None:
        return np.nan

    loader = make_test_loader(modulation_name, num_pilots, snr_db)
    return defined_evaluate(model, loader, num_pilots, device)


def evaluate_methods(modulation_name, num_pilots, snr_db, device):
    mmse_ser = evaluate_mmse(
        num_blocks=TEST_BLOCKS,
        block_length=BLOCK_LENGTH,
        modulation_name=modulation_name,
        snr_db=snr_db,
        num_pilots=num_pilots,
        seed=TEST_SEED,
    )

    icl_ser = evaluate_icl_checkpoint(
        modulation_name,
        num_pilots,
        snr_db,
        device,
    )

    defined_ser = evaluate_defined_checkpoint(
        modulation_name,
        num_pilots,
        snr_db,
        device,
    )

    print(
        f"{modulation_name:>5s} | k={num_pilots} | SNR={snr_db:>2} dB | "
        f"MMSE={mmse_ser:.4f} | ICL={icl_ser:.4f} | DEFINED={defined_ser:.4f}"
    )

    return {
        "MMSE": mmse_ser,
        "ICL": icl_ser,
        "DEFINED": defined_ser,
    }


def save_line_plot(x_values, series, xlabel, ylabel, title, output_name):
    plt.figure(figsize=(7, 4.5))

    for label, y_values in series.items():
        plt.plot(x_values, y_values, marker="o", linewidth=2, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    output_path = PLOT_DIR / output_name
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved {output_path}")


def save_grouped_bar_plot(categories, series, ylabel, title, output_name):
    labels = list(series.keys())
    x = np.arange(len(categories))
    width = 0.8 / len(labels)

    plt.figure(figsize=(8, 4.5))

    for i, label in enumerate(labels):
        offset = (i - (len(labels) - 1) / 2) * width
        plt.bar(x + offset, series[label], width=width, label=label)

    plt.xticks(x, categories)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    output_path = PLOT_DIR / output_name
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved {output_path}")


def experiment_limited_pilots(device):
    print("\n=== Experiment 1: Limited Pilot Detection ===")

    results = {
        "MMSE": [],
        "ICL": [],
        "DEFINED": [],
    }

    for k in PILOT_COUNTS:
        ser = evaluate_methods(
            PILOT_SWEEP_MODULATION,
            k,
            FIXED_SNR_DB,
            device,
        )

        for method in results:
            results[method].append(ser[method])

    save_line_plot(
        PILOT_COUNTS,
        results,
        xlabel="Number of Pilots (k)",
        ylabel="Symbol Error Rate",
        title=f"Limited Pilot Detection ({PILOT_SWEEP_MODULATION}, SNR={FIXED_SNR_DB} dB)",
        output_name="experiment1_limited_pilots.png",
    )

    return results


def experiment_decision_feedback_gain(pilot_results):
    print("\n=== Experiment 2: Decision Feedback Impact ===")

    icl = np.asarray(pilot_results["ICL"], dtype=float)
    defined = np.asarray(pilot_results["DEFINED"], dtype=float)

    gain = 100.0 * (icl - defined) / np.maximum(icl, 1e-12)

    for k, value in zip(PILOT_COUNTS, gain):
        print(f"k={k} | DF gain over ICL: {value:.2f}%")

    save_line_plot(
        PILOT_COUNTS,
        {"DF gain (%)": gain},
        xlabel="Number of Pilots (k)",
        ylabel="SER Improvement Over ICL (%)",
        title=f"Decision Feedback Impact ({PILOT_SWEEP_MODULATION}, SNR={FIXED_SNR_DB} dB)",
        output_name="experiment2_decision_feedback_gain.png",
    )


def experiment_modulation_complexity(device):
    print("\n=== Experiment 3: Modulation Complexity ===")

    results = {
        "MMSE": [],
        "ICL": [],
        "DEFINED": [],
    }

    for modulation_name in MODULATIONS:
        ser = evaluate_methods(
            modulation_name,
            MODULATION_SWEEP_PILOTS,
            FIXED_SNR_DB,
            device,
        )

        for method in results:
            results[method].append(ser[method])

    save_grouped_bar_plot(
        MODULATIONS,
        results,
        ylabel="Symbol Error Rate",
        title=f"Modulation Complexity (k={MODULATION_SWEEP_PILOTS}, SNR={FIXED_SNR_DB} dB)",
        output_name="experiment3_modulation_complexity.png",
    )


def experiment_snr_sensitivity(device):
    print("\n=== Experiment 4: SNR Sensitivity ===")

    results = {
        "MMSE": [],
        "ICL": [],
        "DEFINED": [],
    }

    for snr_db in SNR_VALUES:
        ser = evaluate_methods(
            PILOT_SWEEP_MODULATION,
            MODULATION_SWEEP_PILOTS,
            snr_db,
            device,
        )

        for method in results:
            results[method].append(ser[method])

    save_line_plot(
        SNR_VALUES,
        results,
        xlabel="SNR (dB)",
        ylabel="Symbol Error Rate",
        title=f"SNR Sensitivity ({PILOT_SWEEP_MODULATION}, k={MODULATION_SWEEP_PILOTS})",
        output_name="experiment4_snr_sensitivity.png",
    )


def mmse_ser_by_position(modulation_name, num_pilots, snr_db):
    rng = np.random.default_rng(TEST_SEED)
    constellation = get_constellation(modulation_name)

    errors = np.zeros(BLOCK_LENGTH, dtype=np.float64)

    for _ in range(TEST_BLOCKS):
        block = generate_block(
            block_length=BLOCK_LENGTH,
            modulation_name=modulation_name,
            snr_db=snr_db,
            r=rng,
        )

        x = block["transmitted_symbols"]
        y = block["received_symbols"]
        labels = block["transmitted_labels"]
        noise_variance = block["noise_variance"]

        h_hat = mmse_estimate(
            x[:num_pilots],
            y[:num_pilots],
            noise_variance,
        )

        for t in range(num_pilots, BLOCK_LENGTH):
            pred_label = detect_symbol(y[t], h_hat, constellation)
            errors[t] += pred_label != labels[t]

    return errors[num_pilots:] / TEST_BLOCKS


def icl_ser_by_position(model, dataloader, num_pilots, device):
    model.eval()

    position_errors = None
    total_blocks = 0

    with torch.no_grad():
        for y_batch, x_batch, labels_batch in dataloader:
            y_batch = y_batch.to(device)
            x_batch = x_batch.to(device)
            labels_batch = labels_batch.to(device)

            batch_size, block_length, _ = x_batch.shape

            if position_errors is None:
                position_errors = torch.zeros(block_length, device=device)

            for t in range(num_pilots, block_length):
                y_prompt = torch.zeros_like(y_batch)
                x_prompt = torch.zeros_like(x_batch)

                y_prompt[:, :num_pilots, :] = y_batch[:, :num_pilots, :]
                x_prompt[:, :num_pilots, :] = x_batch[:, :num_pilots, :]

                y_prompt[:, t, :] = y_batch[:, t, :]

                logits = model(y_prompt, x_prompt)
                preds = torch.argmax(logits[:, t, :], dim=-1)

                position_errors[t] += (preds != labels_batch[:, t]).sum()

            total_blocks += batch_size

    return (position_errors[num_pilots:] / total_blocks).cpu().numpy()


def defined_ser_by_position(model, dataloader, num_pilots, device):
    model.eval()

    position_errors = None
    total_blocks = 0

    with torch.no_grad():
        for y_batch, x_batch, labels_batch in dataloader:
            y_batch = y_batch.to(device)
            x_batch = x_batch.to(device)
            labels_batch = labels_batch.to(device)

            batch_size, block_length, num_classes = x_batch.shape

            if position_errors is None:
                position_errors = torch.zeros(block_length, device=device)

            x_feedback = torch.zeros_like(x_batch)

            for t in range(block_length):
                if t < num_pilots:
                    x_feedback[:, t, :] = x_batch[:, t, :]
                else:
                    logits = model(y_batch, x_feedback)
                    preds = torch.argmax(logits[:, t, :], dim=-1)

                    x_feedback[:, t, :] = torch.nn.functional.one_hot(
                        preds,
                        num_classes=num_classes,
                    ).float()

                    position_errors[t] += (preds != labels_batch[:, t]).sum()

            total_blocks += batch_size

    return (position_errors[num_pilots:] / total_blocks).cpu().numpy()


def experiment_context_position(device):
    print("\n=== Optional: SER by Context Sequence Length ===")

    modulation_name = "QPSK"
    num_pilots = 1
    snr_db = FIXED_SNR_DB

    loader = make_test_loader(
        modulation_name,
        num_pilots,
        snr_db,
    )

    icl_model = load_trained_model(
        "icl",
        modulation_name,
        num_pilots,
        device,
    )

    defined_model = load_trained_model(
        "defined",
        modulation_name,
        num_pilots,
        device,
    )

    if icl_model is None or defined_model is None:
        return

    context_lengths = list(range(num_pilots + 1, BLOCK_LENGTH + 1))

    results = {
        "MMSE": mmse_ser_by_position(
            modulation_name,
            num_pilots,
            snr_db,
        ),
        "ICL": icl_ser_by_position(
            icl_model,
            loader,
            num_pilots,
            device,
        ),
        "DEFINED": defined_ser_by_position(
            defined_model,
            loader,
            num_pilots,
            device,
        ),
    }

    save_line_plot(
        context_lengths,
        results,
        xlabel="Context Sequence Length",
        ylabel="Symbol Error Rate",
        title=f"SER by Context Length ({modulation_name}, k={num_pilots}, SNR={snr_db} dB)",
        output_name="optional_context_sequence_length.png",
    )


def main():
    PLOT_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pilot_results = experiment_limited_pilots(device)

    experiment_decision_feedback_gain(pilot_results)
    experiment_modulation_complexity(device)
    experiment_snr_sensitivity(device)
    experiment_context_position(device)


if __name__ == "__main__":
    main()
