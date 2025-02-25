import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import AttackingObjectiveBase
from utils.registry import ATTACKING_OBJECTIVE
from collections import OrderedDict

@ATTACKING_OBJECTIVE.register()
class DIA(AttackingObjectiveBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rand_init = False

    def attack_objective(self, x, y):
        bs = x.shape[0]
        loss = F.cross_entropy(x[:int(bs / 2)], y[:int(bs / 2)])
        return loss
    
    def hook_before_attack(self, attacked_model, clean_data, **kwargs):
        self.config_network(attacked_model, attack_hook=True)
        return
    
    def hook_after_attack(self, attacked_model, **kwargs):
        self.config_network(attacked_model, attack_hook=False)
        return
    
    def config_network(self, model: torch.nn.Module, attack_hook: bool = True):
        if hasattr(model, 'conv1'):
            model.conv1._backward_hooks = OrderedDict()
            if attack_hook:
                model.conv1.register_full_backward_hook(self.backward_hook)
        elif hasattr(model, 'conv_1_3x3'):
            model.conv_1_3x3._backward_hooks = OrderedDict()
            if attack_hook:
                model.conv_1_3x3.register_full_backward_hook(self.backward_hook)
        elif hasattr(model,'model') and hasattr(model.model, 'conv1'):
            model.model.conv1._backward_hooks = OrderedDict()
            if attack_hook:
                model.model.conv1.register_full_backward_hook(self.backward_hook)
        else:
            raise NotImplementedError
        return

    def backward_hook(self, module, gin, gout):
        bs = gin[0].shape[0]
        gin[0][:int(bs / 2), ...] = 0.
        return gin
