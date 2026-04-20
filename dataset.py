import numpy as np

def build_icl_samples(block, num_pilots):
    
    x = block["transmitted_symbols"]
    y = block["received_symbols"]
    labels = block["transmitted_labels"]

    inputs = []
    targets = []

    T = len(x)

    for t in range(num_pilots, T):
        sequence = []

        for i in range(num_pilots):
            sequence.append(y[i])
            sequence.append(x[i])

        sequence.append(y[t])

        inputs.append(np.array(sequence))
        targets.append(labels[t])

    return inputs, targets


def complex_to_real(sequence):
    real_sequence = []

    for c in sequence:
        real_sequence.append([np.real(c), np.imag(c)])

    return np.array(real_sequence)

from channel import generate_block


def build_dataset(
    num_blocks,
    block_length,
    modulation_name,
    snr_db,
    num_pilots
):
    rng = np.random.default_rng()

    X = []
    y = []

    for _ in range(num_blocks):
        block = generate_block(
            block_length=block_length,
            modulation_name=modulation_name,
            snr_db=snr_db,
            r=rng
        )

        inputs, targets = build_icl_samples(block, num_pilots)

        for seq, label in zip(inputs, targets):
            X.append(complex_to_real(seq))
            y.append(label)

    return X, y



