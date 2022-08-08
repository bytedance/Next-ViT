#!/bin/bash
set -x
GPUS=$1

exp_name=nextvit_cls_exp
time=$(date "+%m%d_%H%M%S")
save_root_dir=../${exp_name}/${time}

if [ ! -d ${save_root_dir} ]; then
    mkdir -p ${save_root_dir}
    echo save root dir is ${save_root_dir}.
else
    echo Error, save root dir ${save_roort_dir} exist, please run the shell again!
    exit 1
fi


python3 -m torch.distributed.launch --nproc_per_node=$GPUS --use_env main.py \
--output-dir ${save_root_dir} \
--dist-eval ${@:2}

