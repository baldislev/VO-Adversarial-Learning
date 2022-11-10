#!/bin/bash

###
# CS236781: Deep Learning
# py-sbatch.sh
#
# This script runs python from within our conda env as a slurm batch job.
# All arguments passed to this script are passed directly to the python
# interpreter.
#

###
# Example usage:
#
# Running the prepare-submission command from main.py as a batch job
# ./py-sbatch.sh main.py prepare-submission --id 123456789
#
# Running all notebooks without preparing a submission
# ./py-sbatch.sh main.py run-nb *.ipynb
#
# Running any other python script myscript.py with arguments
# ./py-sbatch.sh myscript.py --arg1 --arg2=val2
#

###
# Parameters for sbatch
#
NUM_NODES=1
NUM_CORES=2
NUM_GPUS=1
JOB_NAME="PGD_SCHED"
MAIL_USER="ayalaavital9@gmail.com"
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=pytorch-cupy-3

###
# Parameters for the run:
SEED=42
TESTDIR="VO_adv_project_train_dataset_8_frames"
MODEL="tartanvo_1914.pkl"
WORKERS=1
TRAJ_LEN=8

EPOCHS=20
ATTACK="pgd"

T_CRIT="mean_partial_rms"
T_FACTOR=1
T_TARGET_CRIT="patch"
T_TARGET_FACTOR=1
ROT_CRIT="quat_product"
ROT_FACTOR=500
FLOW_CRIT="mse"
FLOW_FACTOR=0

ALPHA=0.05

sbatch \
	-N $NUM_NODES \
	-c $NUM_CORES \
	--gres=gpu:$NUM_GPUS \
	--job-name $JOB_NAME \
	--mail-user $MAIL_USER \
	--mail-type $MAIL_TYPE \
	-o 'slurm-%N-%j.out' \
	-w lambda4 \
<<EOF
#!/bin/bash
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"
# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV
# Run python with the args to the script
python run_attacks.py \
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
	--alpha $ALPHA \
	--attack_t_crit $T_CRIT \
	--attack_t_factor $T_FACTOR \
	--attack_target_t_crit $T_TARGET_CRIT \
	--attack_target_t_factor $T_TARGET_FACTOR \
	--attack_flow_crit $FLOW_CRIT \
	--attack_flow_factor $FLOW_FACTOR \
	--attack_rot_crit $ROT_CRIT \
	--attack_rot_factor $ROT_FACTOR
python run_attacks.py \
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
	--alpha $ALPHA \
	--attack_t_crit $T_CRIT \
	--attack_t_factor $T_FACTOR \
	--attack_target_t_crit $T_TARGET_CRIT \
	--attack_target_t_factor $T_TARGET_FACTOR \
	--attack_flow_crit $FLOW_CRIT \
	--attack_flow_factor $FLOW_FACTOR \
	--attack_rot_crit $ROT_CRIT \
	--attack_rot_factor $ROT_FACTOR \
    --minibatch_size 1
python run_attacks.py \
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
	--alpha $ALPHA \
	--attack_t_crit $T_CRIT \
	--attack_t_factor $T_FACTOR \
	--attack_target_t_crit $T_TARGET_CRIT \
	--attack_target_t_factor $T_TARGET_FACTOR \
	--attack_flow_crit $FLOW_CRIT \
	--attack_flow_factor $FLOW_FACTOR \
	--attack_rot_crit $ROT_CRIT \
	--attack_rot_factor $ROT_FACTOR \
    --minibatch_size 2
python run_attacks.py \
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
	--alpha $ALPHA \
	--attack_t_crit $T_CRIT \
	--attack_t_factor $T_FACTOR \
	--attack_target_t_crit $T_TARGET_CRIT \
	--attack_target_t_factor $T_TARGET_FACTOR \
	--attack_flow_crit $FLOW_CRIT \
	--attack_flow_factor $FLOW_FACTOR \
	--attack_rot_crit $ROT_CRIT \
	--attack_rot_factor $ROT_FACTOR \
    --minibatch_size 5
echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF