import os
import random

import numpy as np
import torch
from transformers import set_seed as set_seed_trf
from accelerate.utils import set_seed as set_seed_acc

from transformers import enable_full_determinism

def set_seed(seed, full_determinism=False):
    print(f"Setting seed to {seed}")
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    set_seed_acc(seed)
    set_seed_trf(seed)

    if full_determinism:
        enable_full_determinism(seed)
