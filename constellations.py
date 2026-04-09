import numpy as np

def get_constellation(modulation_name: str) -> np.ndarray:
    """Return normalized complex constellation points in label order."""
    if modulation_name == "BPSK":
        return np.array([-1, 1])
    elif modulation_name == "QPSK":
        return np.array([-1-1j, -1+1j, 1-1j, 1+1j]) / np.sqrt(2)
    elif modulation_name == "16QAM":
        return np.array([-3-3j, -3-1j, -3+1j, -3+3j,
                         -1-3j, -1-1j, -1+1j, -1+3j,
                          1-3j,  1-1j,  1+1j,  1+3j,
                          3-3j,  3-1j,  3+1j,  3+3j]) / np.sqrt(10)
    elif modulation_name == "64QAM":
        return np.array([-7-7j, -7-5j, -7-3j, -7-1j, -7+1j, -7+3j, -7+5j, -7+7j,
                         -5-7j, -5-5j, -5-3j, -5-1j, -5+1j, -5+3j, -5+5j, -5+7j,
                         -3-7j, -3-5j, -3-3j, -3-1j, -3+1j, -3+3j, -3+5j, -3+7j,
                         -1-7j, -1-5j, -1-3j, -1-1j, -1+1j, -1+3j, -1+5j, -1+7j,
                          1-7j,  1-5j,  1-3j,  1-1j,  1+1j,  1+3j,  1+5j,  1+7j,
                          3-7j,  3-5j,  3-3j,  3-1j,  3+1j,  3+3j,  3+5j,  3+7j,
                          5-7j,  5-5j,  5-3j,  5-1j,  5+1j,  5+3j,  5+5j,  5+7j,
                          7-7j,  7-5j,  7-3j,  7-1j,  7+1j,  7+3j,  7+5j,  7+7j]) / np.sqrt(42)
    else:
        raise ValueError("Unknown modulation scheme.")

def get_constellation_size(modulation_name: str) -> int:
    """Return number of symbols in the constellation."""
    if modulation_name == "BPSK":
        return 2
    elif modulation_name == "QPSK":
        return 4
    elif modulation_name == "16QAM":
        return 16
    elif modulation_name == "64QAM":
        return 64
    else:
        raise ValueError("Unknown modulation scheme.")

def labels_to_symbols(label_array: np.ndarray, modulation_name: str) -> np.ndarray:
    """Convert integer class labels to complex constellation symbols."""
    constellation = get_constellation(modulation_name)
    return constellation[label_array]

def symbols_to_nearest_labels(symbol_array: np.ndarray, modulation_name: str) -> np.ndarray:
    """Quantize complex symbols to the nearest constellation label."""
    constellation = get_constellation(modulation_name)
    distances = np.abs(symbol_array[:, None] - constellation[None, :])
    return np.argmin(distances, axis=1)