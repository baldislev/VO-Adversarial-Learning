#!/bin/bash

###
# Parameters for the run:
SEED=42
TESTDIR="unseen"
MODEL="tartanvo_1914.pkl"
WORKERS=1
TRAJ_LEN=8

EPOCHS=10
NORM='Linf'
ATTACK="const"
MINIBATCH=1
SPLIT="3_4_0_1_2"

T_CRIT="mean_partial_rms"
T_FACTOR=1
T_TARGET_CRIT="patch"
T_TARGET_FACTOR=1
ROT_CRIT="quat_product"
ROT_FACTOR=1
FLOW_CRIT="mse"
FLOW_FACTOR=1

MOMENTUM=0.1
RMSPROP=0.7
ALPHA=0.05
BETA=0.5
LOAD_PERT="./results/kitti_custom/tartanvo_1914/VO_adv_project_train_dataset_8_frames/train_attack/universal_attack/stochastic_gradient_ascent_minibatch_size_1/split_data_3_4_0_1_2/attack_pgd_norm_Linf/opt_whole_trajectory/opt_t_crit_mean_partial_rms_factor_1_0_rot_crit_quat_product_factor_1_0_flow_crit_mse_factor_1_0_target_t_crit_patch_factor_1_0/eval_rms/eps_1_attack_iter_10_alpha_0_05_t_decay_None/adv_best_pert/adv_best_pert.png"

srun -c 2 --gres=gpu:1 --pty python run_attacks.py \
	--save_imgs \
	--save_best_pert \
	--preprocessed_data \
	--save_csv \
	--seed $SEED \
	--test-dir $TESTDIR \
	--model-name $MODEL \
	--max_traj_len $TRAJ_LEN \
	--worker-num $WORKERS \
	--attack $ATTACK \
	--attack_k $EPOCHS \
	--attack_norm $NORM \
	--load_attack $LOAD_PERT
