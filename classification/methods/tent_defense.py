import math
import torch.nn as nn
import torch.jit

from copy import deepcopy
from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy

from utils.bn_layers import RobustBN2d, RobustMedianBN2d
from augmentations.transforms_cotta import get_tta_transforms

@ADAPTATION_REGISTRY.register()
class Tent_D(TTAMethod):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        # setup loss function
        self.softmax_entropy = Entropy()

        if self.cfg.TENT_D.EMA_MODEL:
            self.model_ema = self.copy_model(self.model)
            for param in self.model_ema.parameters():
                param.detach_()
        else:
            self.model_ema = self.model
        
        if self.cfg.TENT_D.ENSEMBLE_SOURCE_PARAM:
            self.src_model = deepcopy(self.model)
            for param in self.src_model.parameters():
                param.detach_()

        if self.cfg.TENT_D.DATA_AUGMENTATION:
            self.transform = get_tta_transforms(self.dataset_name)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        self.model.train()
        self.model_ema.train()
        
        imgs_test = x[0]
        if not self.cfg.TENT_D.DATA_AUGMENTATION and not self.cfg.TENT_D.EMA_MODEL:
            outputs = self.model(imgs_test)
        else:
            outputs = self.model_ema(imgs_test)
        
        if self.cfg.TENT_D.ENTROPY_THRESHOLD:
            mask = softmax_cross_entropy(outputs, outputs) < math.log(self.num_classes) * self.cfg.TENT_D.THRESHOLD_VALUE

        if self.cfg.TENT_D.DATA_AUGMENTATION:
            aug_output = self.model(self.transform(imgs_test))
            if self.cfg.TENT_D.ENTROPY_THRESHOLD:
                loss = softmax_cross_entropy(aug_output, outputs)[mask].mean()
            else:
                loss = softmax_cross_entropy(aug_output, outputs).mean()
        elif self.cfg.TENT_D.EMA_MODEL:
            online_output = self.model(imgs_test)
            if self.cfg.TENT_D.ENTROPY_THRESHOLD:
                loss = softmax_cross_entropy(online_output, outputs)[mask].mean()
            else:
                loss = softmax_cross_entropy(online_output, outputs).mean()
        else:
            if self.cfg.TENT_D.ENTROPY_THRESHOLD:
                loss = self.softmax_entropy(outputs)[mask].mean()
            else:
                loss = self.softmax_entropy(outputs).mean()

        self.optimizer.zero_grad()
        if not self.cfg.TENT_D.ENTROPY_THRESHOLD or mask.sum() > 0:
            loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        with torch.no_grad():
            if self.cfg.TENT_D.ENSEMBLE_SOURCE_PARAM:
                self.update_ema_variables(self.model, self.src_model, self.cfg.TENT_D.ENSEMBLE_MOMENTUM)

            if self.cfg.TENT_D.EMA_MODEL:
                self.model_ema = self.update_ema_variables(self.model_ema, self.model, self.cfg.TENT_D.MODEL_MOMENTUM)

        return outputs
    
    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model

    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.

        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, RobustBN2d, RobustMedianBN2d)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        # self.model.train()
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what tent updates
        self.model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        for name, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                
                if self.cfg.TENT_D.M_DIA or self.cfg.TENT_D.MEDIA_BN:
                    if self.cfg.TENT_D.M_DIA:
                        New_BN = RobustBN2d
                    elif self.cfg.TENT_D.MEDIA_BN:
                        New_BN = RobustMedianBN2d
                    
                    momentum_bn = New_BN(m, self.cfg.TENT_D.BN_MOMENTUM)
                    momentum_bn.requires_grad_(True)
                    set_named_submodule(self.model, name, momentum_bn)
                else:
                    m.requires_grad_(True)
                    # force use of batch stats in train and eval modes
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)


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

@torch.jit.script
def softmax_cross_entropy(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)
