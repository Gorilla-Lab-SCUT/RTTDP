"""
Builds upon: https://github.com/Gorilla-Lab-SCUT/TRIBE
Corresponding paper: https://arxiv.org/abs/2309.14949
"""
import math
import torch
import torch.nn as nn
import torch.jit

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy
from augmentations.transforms_cotta import get_tta_transforms
from utils.bn_layers import BalancedRobustBN2dV5, BalancedRobustBN2dEMA, BalancedRobustBN1dV5


@ADAPTATION_REGISTRY.register()
class TRIBE(TTAMethod):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.aux_model = self.copy_model(self.model)
        for param in self.aux_model.parameters():
            param.detach_()

        for (name1, param1), (name2, param2) in zip(self.model.named_parameters(), self.aux_model.named_parameters()):
            set_named_submodule(self.aux_model, name2, param1)

        self.source_model = self.copy_model(self.model)
        for param in self.source_model.parameters():
            param.detach_()

        self.transform = get_tta_transforms(self.dataset_name)

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data):
        x = batch_data[0]
        with torch.no_grad():
            self.aux_model.eval()
            ema_out = self.aux_model(x)

        self.update_model(x, ema_out)

        return ema_out
    
    def update_model(self, batch_data, logit):
        p_l = logit.argmax(dim=1)
        self.source_model.train()
        self.aux_model.train()
        self.model.train()
        strong_sup_aug = self.transform(batch_data)
        
        self.set_bn_label(self.aux_model, p_l)
        ema_sup_out = self.aux_model(batch_data)

        self.set_bn_label(self.model, p_l)
        stu_sup_out = self.model(strong_sup_aug)

        entropy = self.self_softmax_entropy(ema_sup_out)
        entropy_mask = (entropy < self.cfg.TRIBE.H0 * math.log(self.num_classes))

        l_sup = torch.nn.functional.cross_entropy(stu_sup_out, ema_sup_out.argmax(dim=-1), reduction='none')[entropy_mask].mean()

        with torch.no_grad():
            self.set_bn_label(self.source_model, p_l)
            source_anchor = self.source_model(batch_data).detach()
        
        l_reg = self.cfg.TRIBE.LAMBDA * torch.nn.functional.mse_loss(ema_sup_out, source_anchor, reduction='none')[entropy_mask].mean()

        l = (l_sup + l_reg)

        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()

        return

    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.

        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []

        for n, p in self.model.named_parameters():
            if p.requires_grad:
                names.append(n)
                params.append(p)
        return params, names

    @staticmethod
    def set_bn_label(model, label=None):
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, BalancedRobustBN1dV5) or isinstance(sub_module, BalancedRobustBN2dV5) or isinstance(sub_module, BalancedRobustBN2dEMA):
                sub_module.label = label
        return
    
    @staticmethod
    def self_softmax_entropy(x):
        return -(x.softmax(dim=-1) * x.log_softmax(dim=-1)).sum(dim=-1)

    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model
    
    def configure_model(self):
        self.model.requires_grad_(False)
        normlayer_names = []

        for name, sub_module in self.model.named_modules():
            if isinstance(sub_module, nn.BatchNorm2d) or isinstance(sub_module, nn.BatchNorm1d):
                normlayer_names.append(name)
                
        for name in normlayer_names:
            bn_layer = get_named_submodule(self.model, name)
            if isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = BalancedRobustBN2dV5
            elif isinstance(bn_layer, nn.BatchNorm1d):
                NewBN = BalancedRobustBN1dV5
            else:
                raise RuntimeError()
            
            momentum_bn = NewBN(bn_layer,
                                self.num_classes,
                                self.cfg.TRIBE.ETA,
                                self.cfg.TRIBE.GAMMA
                                )
            momentum_bn.requires_grad_(True)
            set_named_submodule(self.model, name, momentum_bn)



def get_named_submodule(model, sub_name: str):
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)

    return module


def set_named_submodule(model, sub_name, value):
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])

        else:
            setattr(module, names[i], value)

