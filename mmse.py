import numpy as np
from channel import generate_block
from constellations import get_constellation


def mmse_estimate(pilot_x, pilot_y, noise_variance):
    X = pilot_x.reshape(1, -1)
    Y = pilot_y.reshape(1, -1)

    H_hat = Y @ np.conj(X.T) @ np.linalg.inv(X @ np.conj(X.T) + noise_variance)
    return H_hat[0, 0]

def detect_symbol(y, H_hat, constellation):
    candidates = H_hat * constellation
    distances = np.abs(candidates - y)
    return np.argmin(distances)


def evaluate_mmse(num_blocks, block_length, modulation_name,
                  snr_db, num_pilots, seed=123):
    
    rng = np.random.default_rng(seed)
    constellation = get_constellation(modulation_name)

    total_errors = 0
    total_symbols = 0

    for _ in range(num_blocks):
        block = generate_block(
            block_length=block_length,
            modulation_name=modulation_name,
            snr_db=snr_db,
            r=rng,
        )

        x = block["transmitted_symbols"]
        y = block["received_symbols"]
        labels = block["transmitted_labels"]
        noise_variance = block["noise_variance"]

        pilot_x = x[:num_pilots]
        pilot_y = y[:num_pilots]

        h_hat = mmse_estimate(pilot_x, pilot_y, noise_variance)

        for t in range(num_pilots, block_length):
            pred_label = detect_symbol(y[t], h_hat, constellation)

            if pred_label != labels[t]:
                total_errors += 1

            total_symbols += 1

    return total_errors / total_symbols