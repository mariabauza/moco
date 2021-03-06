#!/bin/bash


#Slurm sbatch options
#SBATCH -a 0-3
#SBATCH -c 40
#SBATCH -o logs/test_4_nov_k=20_v2_%j_%a.log
#SBATCH --gres=gpu:volta:1


echo 'TODOOOOOOOOOOOOOOOOOOOO: remember to also change date in run_model'

# Loading the modules
source /etc/profile
module load cuda/10.0


# Run the script
echo $SLURM_ARRAY_TASK_ID
echo $GPU_DEVICE_ORDINAL
export EGL_DEVICE_ID=$GPU_DEVICE_ORDINAL

#export SLURM_ENV_NAME=dclaw_turn #cheetah
#export SLURM_LOG_NAME=22-10_$SLURM_ENV_NAME\_small_lr_big_batch_state_indep
python run_single.py
