import os
import numpy as np
import torch

# Must be set before any CUDA context is created (i.e. before the first
# tensor is moved to a GPU) to make cuBLAS matmul/conv ops reproducible.
# No-op on CPU-only runs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

class RunningNormalizer:
    """Tracks running mean and standard deviation for reward normalization."""
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.M2 = 0.0
    
    def update(self, x):
        # Welford's online algorithm
        if isinstance(x, np.ndarray):
            for val in x.flatten():
                self._update_single(val)
        else:
            self._update_single(x)
            
    def _update_single(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        if self.count > 1:
            self.var = self.M2 / (self.count - 1)
    
    def normalize(self, x):
        if self.count < 2 or self.var <= 0:
            return x
        std = np.sqrt(self.var)
        # Scale protection: divide by max(1.0, std) to avoid scaling up tiny variances
        scale = max(1.0, std)
        return (x - self.mean) / scale

def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # GPU determinism: cuDNN's autotuner (benchmark mode) picks algorithms
    # via non-deterministic timing races, and cuDNN conv/RNN kernels are
    # non-deterministic by default. These flags are inert on CPU-only runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # warn_only=True: use a deterministic kernel wherever one exists, but
    # warn (rather than raise) for any op that has none, so an unrelated
    # future op added elsewhere can't turn this into a hard crash.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Older torch versions lack the warn_only kwarg.
        torch.use_deterministic_algorithms(True)

def moving_average(data, window_size=100):
    if len(data) < window_size:
        return np.array(data)
    return np.convolve(data, np.ones(window_size), 'valid') / window_size
