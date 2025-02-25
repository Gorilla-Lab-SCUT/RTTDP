import torch
import torch.nn as nn

class AttackingObjectiveBase(nn.Module):
    def __init__(self, cfg, attack_algo):
        super().__init__()
        self.cfg = cfg
        self.attack_algo = attack_algo
        self.rand_init=True
    
    def hook_before_attack(self, attacked_model, clean_data, **kwargs):
        return
    
    def hook_after_attack(self, attacked_model, **kwargs):
        return

    def hook_when_return_result(self, probs, y):
        return
    
    def attack_objective(self, x, y):
        return 0.
    
    def hook_towards_loss(self, loss):
        return