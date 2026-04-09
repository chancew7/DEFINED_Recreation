def get_constellation(modulation_name: str) -> np.ndarray:
    """Return normalized complex constellation points in label order."""

def get_constellation_size(modulation_name: str) -> int:
    """Return number of symbols in the constellation."""

def labels_to_symbols(label_array: np.ndarray, modulation_name: str) -> np.ndarray:
    """Convert integer class labels to complex constellation symbols."""

def symbols_to_nearest_labels(symbol_array: np.ndarray, modulation_name: str) -> np.ndarray:
    """Quantize complex symbols to the nearest constellation label."""