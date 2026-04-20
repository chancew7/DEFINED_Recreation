import numpy as np
from channel import generate_block

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
            sequence.append((y[i], 0)) 
            sequence.append((x[i], 1)) 

        sequence.append((y[t], 0))

        inputs.append(np.array(sequence))
        targets.append(labels[t])

    return inputs, targets


def complex_to_real(sequence):
    real_sequence = []

    for c, token_type in sequence:
        real_sequence.append([np.real(c), np.imag(c), token_type])

    return np.array(real_sequence)


def build_dataset(
    num_blocks,
    block_length,
    modulation_name,
    snr_db,
    num_pilots
):
    rng = np.random.default_rng(0)

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
            
            seq_real = complex_to_real(seq)
            power = np.mean(seq_real[:, :2] ** 2)
            seq_real[:, :2] = seq_real[:, :2] / np.sqrt(power + 1e-8)
            X.append(seq_real)

            y.append(label)

    return X, y



