#!/bin/bash

function random(){
    for i in $@; do
        echo "$i $RANDOM";
    done | sort -k2n | cut -d " " -f1;
}

function checkGPU(){
    for gpu in $(random $@); do
        gpu_stat=($(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu));
        if [ $gpu_stat -lt 1000 ]
        then
            return $gpu;
        fi;
    done;
    return 255;
}

function sub(){
    result=""
    for a in $1; do
        if echo $2 | grep -wq $a 
        then
            :
        else
            result="$result $a";
        fi
    done;
    echo $result
}

# This is a queue storing which gpus you want to listen and use.
USE_GPU=(0 1 2 3 4 5 6 7);


# ===================================

running_gpu=""
running_pid=""

# tent rpl eata ttac sar cotta roid
for method in tent; do
# SurrogateModelEstimateAttack SurrogateModelPGDAttack NoAttack
# adversarial attack methods (objective: "[]"): SurrogateModelAutoAttack SurrogateModelGMSAMINAttack SurrogateModelUnlearnableAttack SurrogateModelAdvPoisoningAttack SurrogateModelGMSAAvgAttack
    for algo in SurrogateModelEstimateAttack; do
        for ratio in 0.5; do
        # non_uniform_batch batch 
            for dist in batch; do
                # "['NHEAttackingObjective','Distribution_Regularization']" "['BLEAttackingObjective','Distribution_Regularization']" "['DIA']" "['MaxCE']" "['TePA']" 
                for objective in "['NHEAttackingObjective','Distribution_Regularization']"; do
                    for options in ""; do
                        while [ 1 == 1 ];
                        do
                            check_gpu_id=($( sub "$( echo "${USE_GPU[@]}" )" "$( echo "${running_gpu[@]}" )" ))
                            if [ ${#check_gpu_id[@]} -gt 0 ]
                            then
                                checkGPU $( echo "${check_gpu_id[@]}" )
                                avaible_gpu=$?
                                if [ $avaible_gpu != 255 ]
                                then
                                    running_gpu="$running_gpu $avaible_gpu"
                                    echo running_gpu: $running_gpu;
                                    # main task (You need to modify the below line)
                                    bash ./run_one_exp_base.sh $avaible_gpu $method $algo $ratio "${objective} ${options}" $dist
                                    ##
                                    running_pid="$running_pid $!"
                                    echo running_pid: $running_pid;
                                    break;
                                fi;
                            fi
                            
                            idx=0
                            tmp_gpu=$running_gpu;
                            for pid in $running_pid; do
                                kill -s 0 $pid;
                                if [ $? == 1 ]
                                then
                                    stop_gpu=($tmp_gpu);
                                    running_gpu=$( sub "$( echo "${running_gpu[@]}" )" "${stop_gpu[$idx]}" )
                                    running_pid=$( sub "$( echo "${running_pid[@]}" )" "$pid" )
                                fi
                                let idx+=1;
                            done;
                            sleep 30;
                        done;
                        let option_idx+=1;
                    done;
                done;
            done;
        done;
    done;
done;
