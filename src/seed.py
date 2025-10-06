import random
import numpy as np
import torch

def set_seed(seed: int = 55) -> None:
    """Set random seed for reproducibility.

    Args:
        seed (int): The seed value to use. Default is 55.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False