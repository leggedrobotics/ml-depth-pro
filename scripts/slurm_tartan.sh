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

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <src_file> <outdir>"
    exit 1
fi

source ~/.bashrc
conda activate depth-pro

srcdir=/cluster/work/rsl/patelm/imagenet-1k
outdir=$2

cd /cluster/home/patelm/ws/rsl/ml-depth-pro

echo "Cuda visible devices ${CUDA_VISIBLE_DEVICES}"

python scripts/inferece_tartan.py --srcdir $srcdir --outdir $outdir

echo "Finished running the model"