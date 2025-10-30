import torch
from typing import Optional
import os

from modeling.basic.flash_attn import quant_attn_score_triton
from torch.nn import functional as F

from torch.nn.attention import SDPBackend, sdpa_kernel

import math

import numpy as np

