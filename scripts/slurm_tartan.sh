#!/bin/bash

#SBATCH --ntask 1
#SBATCH --time=16:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16G
#SBATCH --job-name=imagenet-depthpro
#SBATCH --output=/cluster/work/rsl/patelm/result/slurm_output/%x_%j.out
#SBATCH --error=/cluster/work/rsl/patelm/result/slurm_output/%x_%j.err

source ~/.bashrc
conda activate depth-pro

env_name=ModularNeighborhood

srcdir=/cluster/scratch/patelm/$env_name
outdir=/cluster/scratch/patelm/ml_depth_pro/$env_name

cd /cluster/home/patelm/ws/rsl/ml-depth-pro

echo "Cuda visible devices ${CUDA_VISIBLE_DEVICES}"

python scripts/inferece_tartan.py --srcdir $srcdir --outdir $outdir

echo "Finished running the model"