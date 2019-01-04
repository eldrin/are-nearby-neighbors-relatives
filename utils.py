import numpy as np


def save_mulaw(path, x, quantization_channel=256):
    """Encode the given signal with Mu-Law quantization and save

    Args:
        path (str): path to save signal after encoding
        x (numpy.ndarray): signal (n_samples,)
        quantization_channel (int): # of quantization channels
    """
    mu = quantization_channel - 1.
    safe_audio_abs = np.minimum(np.abs(x), 1)
    magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
    signal = np.sign(x) * magnitude
    y = ((signal + 1) / 2 * mu + 0.5).astype(np.uint8)
    np.save(path, y)


def load_mulaw(fn, quantization_channel=256):
    """Load a Mu-Law encoded signal by decoding it
    """
    signal = np.load(fn)
    mu = quantization_channel - 1
    signal = 2 * (signal / mu) - 1
    magnitude = (1 / mu) * ((1 + mu)**np.abs(signal) - 1)
    return np.sign(signal) * magnitude
