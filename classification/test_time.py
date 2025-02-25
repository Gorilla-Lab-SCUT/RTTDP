import os
import torch
import logging
import numpy as np
import methods
import poisoning_attacks
from copy import deepcopy

from models.model import get_model
from utils.eval_utils import get_accuracy, eval_domain_dict
from utils.registry import ADAPTATION_REGISTRY, ATTACK_REGISTRY
from datasets.data_loading import get_test_loader
from conf import cfg, load_cfg_from_args, get_num_classes, ckpt_path_to_domain_seq
import setproctitle
setproctitle.setproctitle('TTA')

logger = logging.getLogger(__name__)
logger.info=print

def evaluate(description):
    load_cfg_from_args(description)
    valid_settings = ["reset_each_shift",           # reset the model state after the adaptation to a domain
                      "continual",                  # train on sequence of domain shifts without knowing when a shift occurs
                      "gradual",                    # sequence of gradually increasing / decreasing domain shifts
                      "mixed_domains",              # consecutive test samples are likely to originate from different domains
                      "correlated",                 # sorted by class label
                      "mixed_domains_correlated",   # mixed domains + sorted by class label
                      "gradual_correlated",         # gradual domain shifts + sorted by class label
                      "reset_each_shift_correlated",
                      "continual_poisoning_attack"
                      ]
    assert cfg.SETTING in valid_settings, f"The setting '{cfg.SETTING}' is not supported! Choose from: {valid_settings}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)

    # get the base model and its corresponding input pre-processing (if available)
    base_model, model_preprocess = get_model(cfg, num_classes, device)

    base_model_bk = deepcopy(base_model)

    # append the input pre-processing to the base model
    base_model.model_preprocess = model_preprocess

    # setup test-time adaptation method
    available_adaptations = ADAPTATION_REGISTRY.registered_names()
    assert cfg.MODEL.ADAPTATION in available_adaptations, \
        f"The adaptation '{cfg.MODEL.ADAPTATION}' is not supported! Choose from: {available_adaptations}"
    model = ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION)(cfg=cfg, model=base_model, num_classes=num_classes)
    logger.info(f"Successfully prepared test-time adaptation method: {cfg.MODEL.ADAPTATION}")

    # get the test sequence containing the corruptions or domain names
    if cfg.CORRUPTION.DATASET == "domainnet126":
        # extract the domain sequence for a specific checkpoint.
        domain_sequence = ckpt_path_to_domain_seq(ckpt_path=cfg.MODEL.CKPT_PATH)
    elif cfg.CORRUPTION.DATASET in ["imagenet_d", "imagenet_d109"] and not cfg.CORRUPTION.TYPE[0]:
        # domain_sequence = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        domain_sequence = ["clipart", "infograph", "painting", "real", "sketch"]
    else:
        domain_sequence = cfg.CORRUPTION.TYPE
    logger.info(f"Using the following domain sequence: {domain_sequence}")

    # prevent iterating multiple times over the same data in the mixed_domains setting
    domain_seq_loop = ["mixed"] if "mixed_domains" in cfg.SETTING else domain_sequence

    # setup the severities for the gradual setting
    if "gradual" in cfg.SETTING and cfg.CORRUPTION.DATASET in ["cifar10_c", "cifar100_c", "imagenet_c"] and len(cfg.CORRUPTION.SEVERITY) == 1:
        severities = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        logger.info(f"Using the following severity sequence for each domain: {severities}")
    else:
        severities = cfg.CORRUPTION.SEVERITY

    if 'poisoning_attack' in cfg.SETTING:
        attack_algo = ATTACK_REGISTRY.get(cfg.ATTACK.ALGO)
        if attack_algo.__base__.__name__ == 'AttackBase':
            attack_algo = attack_algo(cfg=cfg)
        elif attack_algo.__base__.__name__ == 'WhiteBoxModelAttackBase':
            if cfg.MODEL.ADAPTATION == 'cotta':
                attack_algo = attack_algo(cfg=cfg, white_box_model=model.model_ema)
            else:
                attack_algo = attack_algo(cfg=cfg, white_box_model=model.model)
            attack_algo.to(device)
        elif attack_algo.__base__.__name__ == 'SurrogateModelAttackBase':
            attack_algo = attack_algo(cfg=cfg, surrogate_model=base_model_bk)
            attack_algo.to(device)
        else:
            raise NotImplementedError
    else:
        attack_algo = None

    errs = []
    class_errs = []
    errs_5 = []
    domain_dict = {}

    # start evaluation
    for i_dom, domain_name in enumerate(domain_seq_loop):
        if i_dom == 0 or "reset_each_shift" in cfg.SETTING:
            try:
                model.reset()
                logger.info("resetting model")
            except AttributeError:
                logger.warning("not resetting model")
        else:
            logger.warning("not resetting model")

        for severity in severities:
            test_data_loader = get_test_loader(setting=cfg.SETTING,
                                               adaptation=cfg.MODEL.ADAPTATION,
                                               dataset_name=cfg.CORRUPTION.DATASET,
                                               preprocess=model_preprocess,
                                               data_root_dir=cfg.DATA_DIR,
                                               domain_name=domain_name,
                                               domain_names_all=domain_sequence,
                                               severity=severity,
                                               num_examples=cfg.CORRUPTION.NUM_EX,
                                               rng_seed=cfg.RNG_SEED,
                                               delta_dirichlet=cfg.TEST.DELTA_DIRICHLET,
                                               batch_size=cfg.TEST.BATCH_SIZE,
                                               shuffle=False,
                                               workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()),
                                               poisoning_ratio=cfg.ATTACK.POISONING_RATIO,
                                               attack_data_type=cfg.ATTACK.DATA_TYPE
                                               )

            # evaluate the model
            acc, domain_dict, num_samples, class_avg_acc = get_accuracy(model,
                                                         data_loader=test_data_loader,
                                                         dataset_name=cfg.CORRUPTION.DATASET,
                                                         domain_name=domain_name,
                                                         setting=cfg.SETTING,
                                                         domain_dict=domain_dict,
                                                         print_every=cfg.PRINT_EVERY,
                                                         device=device,
                                                         attack_algo=attack_algo)

            err = 1. - acc
            errs.append(err)
            class_errs.append(1. - class_avg_acc)
            if severity == 5 and domain_name != "none":
                errs_5.append(err)

            logger.info(f"{cfg.CORRUPTION.DATASET} error % [{domain_name}{severity}][#samples={num_samples}]: {err:.2%}; {class_errs[-1]:.2%}")

    if len(errs_5) > 0:
        logger.info(f"mean error: {np.mean(errs):.2%}, mean error at 5: {np.mean(errs_5):.2%}, mean class error: {np.mean(class_errs):.2%}")
    else:
        logger.info(f"mean error: {np.mean(errs):.2%}, mean class error: {np.mean(class_errs):.2%}")

    if "mixed_domains" in cfg.SETTING and len(domain_dict.values()) > 0:
        # print detailed results for each domain
        eval_domain_dict(domain_dict, domain_seq=domain_sequence)


if __name__ == '__main__':
    evaluate('"Evaluation.')
