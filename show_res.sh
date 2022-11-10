#!/bin/bash

###
# Parameters for the run:
SORT_CRIT="test_adv_delta_rms"
FILTER_K="norm"
FILTER_V="Linf"
PLOT_K="eval_data"
Y_LABEL="test_adv_ratio_rms"
X_LABEL="evaluation folder"
TITLE="Evaluation folder experiment for Momentum PGD" #"Loss experiment for optimal rot_crit=quat_product factor"

srun -c 2 --gres=gpu:1 --pty python experiments/res_parser.py \
	--sort_crit $SORT_CRIT \
	--filter_keys $FILTER_K \
	--filter_vals $FILTER_V \
	--x_keys $PLOT_K \
	--x_label "$X_LABEL" \
	--y_label $Y_LABEL \
	#--title "$TITLE" \
	#--dont_print
