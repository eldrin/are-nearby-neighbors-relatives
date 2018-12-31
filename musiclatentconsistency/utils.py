import numpy as np


def save_mulaw(fn, y, sr=22050, quantization_channel=256):
    """Save a signal encoded with Mu-Law

    Args:
        fn (str): path to save the signal
        y (numpy.ndarray): signal vector (n_samples,)
        sr (int): sampling rate
        quantization_channel (int): # of quantization channels
    """
    mu = quantization_channel - 1
    safe_audio_abs = np.minimum(np.abs(y), 1.)
    magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
    signal = np.sign(y) * magnitude
    y_ = ((signal + 1) / 2 * mu + 0.5).astype(np.uint8)
    np.save(fn, y_)


def load_mulaw(fn, quantization_channel=256):
    """Load a Mu-Law encoded signal by decoding it
    """
    signal = np.load(fn)
    mu = quantization_channel - 1
    signal = 2 * (signal / mu) - 1
    magnitude = (1 / mu) * ((1 + mu)**np.abs(signal) - 1)
    return np.sign(signal) * magnitude
