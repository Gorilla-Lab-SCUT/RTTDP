import torch
import torch.nn as nn
from .base import AttackingObjectiveBase
from utils.registry import ATTACKING_OBJECTIVE

@ATTACKING_OBJECTIVE.register()
class TePA(AttackingObjectiveBase):
    def attack_objective(self, x, y):
        return -(x.softmax(1) * x.log_softmax(1)).sum(1).mean()
