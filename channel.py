import numpy as np

def generate_channel():
    # Generate a random Rayleigh block fading channel
    h = (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)
    return h

def generate_noise(sigma2=1, length=31):
    # Generate Additive White Gaussian Noise (AWGN) with variance sigma2
    n = (np.random.randn(length) + 1j*np.random.randn(length)) * np.sqrt(sigma2/2)
    return n