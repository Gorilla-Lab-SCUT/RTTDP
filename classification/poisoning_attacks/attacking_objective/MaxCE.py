import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import AttackingObjectiveBase
from utils.registry import ATTACKING_OBJECTIVE

import numpy as np

@ATTACKING_OBJECTIVE.register()
class MaxCE(AttackingObjectiveBase):
    def attack_objective(self, x, y):
        return F.cross_entropy(x, y, reduction='none').mean()
