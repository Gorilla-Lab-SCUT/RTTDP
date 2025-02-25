import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import AttackingObjectiveBase
from utils.registry import ATTACKING_OBJECTIVE

@ATTACKING_OBJECTIVE.register()
class NHEAttackingObjective(AttackingObjectiveBase):
    def attack_objective(self, x, y):
        target_y = torch.ones_like(x)
        target_y[torch.arange(y.shape[0]), y] = 0.
        target_y /= target_y.sum(1, keepdim=True)
        return -F.kl_div(x.log_softmax(1), target_y.detach(), reduction='batchmean')
