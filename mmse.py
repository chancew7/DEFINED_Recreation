import numpy as np

def mmse_estimate(pilot_x, pilot_y, noise_variance):
    X = pilot_x.reshape(1, -1)
    Y = pilot_y.reshape(1, -1)

    H_hat = Y @ np.conj(X.T) @ np.linalg.inv(X @ np.conj(X.T) + noise_variance)
    return H_hat[0, 0]

def detect_symbol(y, H_hat, constellation):
    candidates = H_hat * constellation
    distances = np.abs(candidates - y)
    return np.argmin(distances)