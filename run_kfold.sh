#!/bin/bash

script="parameters_train_kfold.py"
args="$(python3.6 $script 0)"
##CUDA_VISIBLE_DEVICES="0,1,2,3" python3.6 main_mod.py --parameters $script $args

script="parameters_eval_kfold.py"
args="$(python3.6 $script 0)"
CUDA_VISIBLE_DEVICES="0,1,2,3" python3.6 main_mod.py --parameters $script $args

script="parameters_train_kfold.py"
args="$(python3.6 $script 1)"
##CUDA_VISIBLE_DEVICES="0,1,2,3" python3.6 main_mod.py --parameters $script $args

script="parameters_eval_kfold.py"
args="$(python3.6 $script 1)"
#CUDA_VISIBLE_DEVICES="0,1,2,3" python3.6 main_mod.py --parameters $script $args

script="parameters_train_kfold.py"
args="$(python3.6 $script 2)"
##CUDA_VISIBLE_DEVICES="0,1,2,3" python3.6 main_mod.py --parameters $script $args

script="parameters_eval_kfold.py"
args="$(python3.6 $script 2)"
#CUDA_VISIBLE_DEVICES="0,1,2,3" python3.6 main_mod.py --parameters $script $args

script="parameters_train_kfold.py"
args="$(python3.6 $script 3)"
##CUDA_VISIBLE_DEVICES="0,1,2,3" python3.6 main_mod.py --parameters $script $args

script="parameters_eval_kfold.py"
args="$(python3.6 $script 3)"
#CUDA_VISIBLE_DEVICES="0,1,2,3" python3.6 main_mod.py --parameters $script $args

script="parameters_train_kfold.py"
args="$(python3.6 $script 4)"
##CUDA_VISIBLE_DEVICES="0,1,2,3" python3.6 main_mod.py --parameters $script $args

script="parameters_eval_kfold.py"
args="$(python3.6 $script 4)"
#CUDA_VISIBLE_DEVICES="0,1,2,3" python3.6 main_mod.py --parameters $script $args


