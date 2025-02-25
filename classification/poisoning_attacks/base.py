import torch
from copy import deepcopy
from utils.registry import ATTACK_REGISTRY


class AttackBase(object):
    def __init__(self, cfg: dict):
        self.cfg = cfg

        self.input_clean_data = None

    def __call__(self, clean_data: torch.Tensor, label: torch.Tensor):
        raise NotImplementedError
    
    def return_results(self, probs: torch.Tensor, y):
        return


class SurrogateModelAttackBase(AttackBase, torch.nn.Module):
    def __init__(self, cfg: dict, surrogate_model: torch.nn.Module):
        AttackBase.__init__(self, cfg)
        torch.nn.Module.__init__(self)
        # avoid surrogate model updated
        self.surrogate_model = deepcopy(surrogate_model)
    
    def __call__(self, clean_data: torch.Tensor, label: torch.Tensor):
        return self.forward(clean_data, label)


class WhiteBoxModelAttackBase(AttackBase, torch.nn.Module):
    def __init__(self, cfg: dict, white_box_model: torch.nn.Module):
        AttackBase.__init__(self, cfg)
        torch.nn.Module.__init__(self)
        # NOTE: inner model is shared with outside model and 
        #       their weights will update together.
        self.white_box_model = white_box_model
    
    def __call__(self, clean_data: torch.Tensor, label: torch.Tensor):
        return self.forward(clean_data, label)

@ATTACK_REGISTRY.register()
class NoAttack(AttackBase):
    def __init__(self, cfg: dict):
        AttackBase.__init__(self, cfg)
    
    def __call__(self, clean_data: torch.Tensor, label: torch.Tensor):
        return self.forward(clean_data, label)
    
    def forward(self, clean_data, label: torch.Tensor):
        return clean_data


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
