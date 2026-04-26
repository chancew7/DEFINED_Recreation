import numpy as np
from channel import generate_block
from constellations import get_constellation, get_constellation_size

def build_icl_samples(block, num_pilots, modulation_name):
    
    x = block["transmitted_symbols"]
    y = block["received_symbols"]
    labels = block["transmitted_labels"]

    constellation   = get_constellation(modulation_name)
    num_classes     = get_constellation_size(modulation_name)

    T = len(x)

    y_seqs, x_seqs, target_labels = [], [], []

    for t in range(num_pilots, T):
        y_seq_complex = np.concatenate([y[:num_pilots], [y[t]]])
        y_seq = np.stack([y_seq_complex.real, y_seq_complex.imag], axis=-1)

        pilot_labels = labels[:num_pilots]
        query_label = labels[t]
        all_labels = np.concatenate([pilot_labels, [query_label]])
        
        x_seq = np.eye(num_classes, dtype=np.float32)[all_labels]

        y_seqs.append(y_seq.astype(np.float32))
        x_seqs.append(x_seq)
        target_labels.append(labels[t])

    return y_seqs, x_seqs, target_labels


def complex_to_real(sequence):
    real_sequence = []

    for c, token_type in sequence:
        real_sequence.append([np.real(c), np.imag(c), token_type])

    return np.array(real_sequence)

def build_icl_samples_eval(block, num_pilots, modulation_name):
    x = block["transmitted_symbols"]
    y = block["received_symbols"]
    labels = block["transmitted_labels"]
    
    num_classes = get_constellation_size(modulation_name)
    T = len(x)  # Full block length (31)
    
    # y: all received symbols
    y_seq = np.stack([y.real, y.imag], axis=-1)  # (T, 2)
    
    # x: all transmitted symbols (one-hot)
    x_seq = np.eye(num_classes, dtype=np.float32)[labels]  # (T, C)
    
    return y_seq, x_seq, labels

def build_dataset(num_blocks, block_length, modulation_name,
                       snr_db_min, snr_db_max, num_pilots, seed=42):
    """Build full-sequence dataset."""
    rng = np.random.default_rng(seed)
    
    all_y, all_x, all_labels = [], [], []
    
    for _ in range(num_blocks):
        snr_db = rng.uniform(snr_db_min, snr_db_max)
        
        block = generate_block(
            block_length=block_length,
            modulation_name=modulation_name,
            snr_db=snr_db,
            r=rng
        )
        
        y_seq, x_seq, labels = build_icl_samples_eval(block, num_pilots, modulation_name)
        
        all_y.append(y_seq)
        all_x.append(x_seq)
        all_labels.append(labels)
    
    return all_y, all_x, all_labels