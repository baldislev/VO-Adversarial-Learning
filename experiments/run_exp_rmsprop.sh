#!/bin/bash

###
# Parameters for the run:
SEED=42
TESTDIR="VO_adv_project_train_dataset_8_frames"
MODEL="tartanvo_1914.pkl"
WORKERS=1
TRAJ_LEN=8

EPOCHS=20
NORM='Linf'
ATTACK="rmsprop"
MINIBATCH=1
#SPLIT="0_1_2_3_4"

T_CRIT="mean_partial_rms"
T_FACTOR=1
T_TARGET_CRIT="patch"
T_TARGET_FACTOR=0.1
ROT_CRIT="quat_product"
ROT_FACTOR=500
FLOW_CRIT="mse"
FLOW_FACTOR=0

MOMENTUM=0.1
RMSPROP=0.7
ALPHA=0.05
BETA=0.5

for RMSPROP in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
		srun -c 2 --gres=gpu:1 -w lambda5 --pty python run_attacks.py \
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
		--alpha $ALPHA \
		--momentum $MOMENTUM \
		--rmsprop_decay $RMSPROP \
		--attack_t_crit $T_CRIT \
		--attack_t_factor $T_FACTOR \
		--attack_target_t_crit $T_TARGET_CRIT \
		--attack_target_t_factor $T_TARGET_FACTOR \
		--attack_rot_crit $ROT_CRIT \
		--attack_rot_factor $ROT_FACTOR
done