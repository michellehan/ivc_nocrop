#!/bin/bash

#args="$(python3.6 parameters_eval.py)"
#CUDA_VISIBLE_DEVICES="0,1" python3.6 main.py $args

script="parameters_eval.py"
args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="0,1" python3.6 main_mod.py --parameters $script $args

script="parameters_eval_groupedB.py"
args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="0,1" python3.6 main_mod.py --parameters $script $args

script="parameters_eval_groupedBC.py"
args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="0,1" python3.6 main_mod.py --parameters $script $args

script="parameters_eval_groupedBG.py"
args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="0,1" python3.6 main_mod.py --parameters $script $args

script="parameters_eval_groupedAll.py"
args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="0,1" python3.6 main_mod.py --parameters $script $args
