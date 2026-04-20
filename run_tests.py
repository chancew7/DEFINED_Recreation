import numpy as np

from channel import generate_block
from constellations import get_constellation
from mmse import mmse_estimate, detect_symbol


def test_channel_equation():
    print("=== Test 1: Channel Equation ===")

    rng = np.random.default_rng(0)

    block = generate_block(
        block_length=5,
        modulation_name="BPSK",
        snr_db=100,
        r=rng
    )

    x = block["transmitted_symbols"]
    y = block["received_symbols"]
    h = block["channel_coefficient"]

    error = np.mean(np.abs(y - h * x))

    print("Mean error:", error)
    print("Expected: near 0\n")


def test_mmse_single_block():
    print("=== Test 2: MMSE Basic ===")

    rng = np.random.default_rng(1)

    block = generate_block(
        block_length=10,
        modulation_name="BPSK",
        snr_db=20,
        r=rng
    )

    constellation = get_constellation("BPSK")

    k = 3

    pilot_x = block["transmitted_symbols"][:k]
    pilot_y = block["received_symbols"][:k]

    H_hat = mmse_estimate(
        pilot_x,
        pilot_y,
        block["noise_variance"]
    )

    print("True H:", block["channel_coefficient"])
    print("Estimated H:", H_hat)
    print()


def test_mmse_detection():
    print("=== Test 3: MMSE Detection ===")

    rng = np.random.default_rng(2)

    block = generate_block(
        block_length=50,
        modulation_name="BPSK",
        snr_db=10,
        r=rng
    )

    constellation = get_constellation("BPSK")

    k = 5  # pilots

    pilot_x = block["transmitted_symbols"][:k]
    pilot_y = block["received_symbols"][:k]

    H_hat = mmse_estimate(
        pilot_x,
        pilot_y,
        block["noise_variance"]
    )

    correct = 0
    total = 0

    for i in range(k, len(block["received_symbols"])):
        y = block["received_symbols"][i]

        pred_label = detect_symbol(y, H_hat, constellation)
        true_label = block["transmitted_labels"][i]

        if pred_label == true_label:
            correct += 1

        total += 1

    ser = 1 - (correct / total)

    print("SER:", ser)
    print("Expected: < 0.1 for BPSK, 10dB\n")


if __name__ == "__main__":
    test_channel_equation()
    test_mmse_single_block()
    test_mmse_detection()