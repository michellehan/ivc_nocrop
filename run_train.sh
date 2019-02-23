#!/bin/bash

script="parameters_train.py"
args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="0,1,2,3" python3.6 main_mod.py --parameters $script $args

script="parameters_train_groupedB.py"
args="$(python3.6 $script)"
#CUDA_VISIBLE_DEVICES="0" python3.6 main_mod.py --parameters $script $args

script="parameters_train_groupedBC.py"
args="$(python3.6 $script)"
#CUDA_VISIBLE_DEVICES="0" python3.6 main_mod.py --parameters $script $args

script="parameters_train_groupedBG.py"
args="$(python3.6 $script)"
#CUDA_VISIBLE_DEVICES="0" python3.6 main_mod.py --parameters $script $args

script="parameters_train_groupedAll.py"
args="$(python3.6 $script)"
#CUDA_VISIBLE_DEVICES="0" python3.6 main_mod.py --parameters $script $args
