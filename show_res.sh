#!/bin/bash

###
# Parameters for the run:
SORT_CRIT='time'
FILTER_K='attack'
FILTER_V='rmsprop_pgd'

srun -c 2 --gres=gpu:1 --pty python experiments/res_parser.py \
	--sort_crit $SORT_CRIT \
	--filter_keys $FILTER_K \
	--filter_vals $FILTER_V
