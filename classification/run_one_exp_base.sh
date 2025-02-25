#!/bin/bash -l

method=$2       # choose from: source, norm_test, tent, rpl, eta, eata, rdump, sar, cotta, rotta, adacontrast, rmt, gtta, lame, roid
setting=continual_poisoning_attack # choose from: continual, mixed_domains, correlated, mixed_domains_correlated, ccc, continual_poisoning_attack
# setting=reset_each_shift
seeds=(1)         # to reproduce the benchmark results, use: (1 2 3 4 5)
options=("ATTACK.DATA_TYPE "$6" ATTACK.POISONING_RATIO "$4" ATTACK.ALGO "$3" ATTACK.OBJECTIVE.NAMES $5 CORRUPTION.SEVERITY [5]")

ccc=("cfgs/ccc/${method}.yaml")
cifar10=("cfgs/cifar10_c/${method}.yaml")
cifar100=("cfgs/cifar100_c/${method}.yaml")
imagenet_c=("cfgs/imagenet_c/${method}.yaml")
imagenet_others=("cfgs/imagenet_others/${method}.yaml")

# ---continual---
for var in "${cifar10[@]}"; do
    for seed in ${seeds[*]}; do
        CUDA_VISIBLE_DEVICES=$1 python test_time.py --cfg $var SETTING $setting RNG_SEED $seed $options
    done
done
