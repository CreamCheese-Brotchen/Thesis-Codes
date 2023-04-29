#!/usr/bin/env bash
#SBATCH --job-name=experiment
#SBATCH --output=~/thesis/logs/experiment%j.log
#SBATCH --error=~/thesis/logs/experiment%j.err
#SBATCH --mail-user=liang@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

source activate toyEnv
# python jobs require the srun command to work
srun python experiment01.py \
                --dataset MNIST \
                --entropy_threshold 0.6 \
                --run_epochs 3 \
                --candidate_start_epoch 0 \
                --tensorboard_comment "MNIST DEBUG" \
                --lr 0.001 \
                --l2 1e-4 \
                --batch_size 16 \
                --not_pretrained \
                --reduce_dataset
#python experiment01.py --dataset MNIST --entropy_threshold 0.6 --run_epochs 3 --candidate_start_epoch 0 --tensorboard_comment "MNIST DEBUG" --lr 0.001 --l2 1e-4 --batch_size 16 --not_pretrained --reduce_dataset