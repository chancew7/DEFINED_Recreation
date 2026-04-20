import numpy as np
from constellations import labels_to_symbols, get_constellation_size

def snr_db_to_noise_variance(snr_db: float) -> float:
    snr_linear = 10 ** (snr_db / 10)
    return 1.0 / snr_linear


def sample_rayleigh_channel(r) -> complex:
    real = r.normal()
    imag = r.normal()
    return (real + 1j * imag) / np.sqrt(2.0)


def sample_complex_gaussian_noise(num_samples: int,
                                  noise_variance: float,
                                  r) -> np.ndarray:
    noise_sd = np.sqrt(noise_variance / 2.0)
    real = r.normal(0.0, noise_sd, num_samples)
    imag = r.normal(0.0, noise_sd, num_samples)
    return real + 1j * imag


def generate_block(block_length: int, 
                    modulation_name: str, 
                    snr_db: float,
                    r) -> dict:
 
    block = {}

    constellation_size = get_constellation_size(modulation_name)
    transmitted_labels = r.integers(low=0,
                                    high=constellation_size,
                                    size=block_length)
    transmitted_symbols = labels_to_symbols(transmitted_labels, modulation_name)
    channel_coefficient = sample_rayleigh_channel(r)
    noise_variance = snr_db_to_noise_variance(snr_db)
    noise_samples = sample_complex_gaussian_noise(block_length, noise_variance, r)

    received_symbols = channel_coefficient * transmitted_symbols + noise_samples

    block = {
        "transmitted_labels": transmitted_labels,
        "transmitted_symbols": transmitted_symbols,
        "received_symbols": received_symbols,
        "channel_coefficient": channel_coefficient,
        "noise_variance": noise_variance,
    }

    return block
    
def generate_dataset(num_blocks: int, 
                    block_length: int,
                    modulation_name: str,
                    snr_db: float) -> dict:
    
    dataset = {}

    transmitted_labels = []
    transmitted_symbols = []
    received_symbols = []
    channel_coefficients = []
    noise_variances = []

    for b in range(num_blocks):
        block_data = generate_block(block_length, modulation_name, snr_db, np.random.default_rng())

        transmitted_labels.append(block_data["transmitted_labels"])
        transmitted_symbols.append(block_data["transmitted_symbols"])
        received_symbols.append(block_data["received_symbols"])
        channel_coefficients.append(block_data["channel_coefficient"])
        noise_variances.append(block_data["noise_variance"])

    dataset = {
        "transmitted_labels": np.stack(transmitted_labels, axis=0),
        "transmitted_symbols": np.stack(transmitted_symbols, axis=0),
        "received_symbols": np.stack(received_symbols, axis=0),
        "channel_coefficients": np.array(channel_coefficients),
        "noise_variances": np.array(noise_variances)
    }
    return dataset
    