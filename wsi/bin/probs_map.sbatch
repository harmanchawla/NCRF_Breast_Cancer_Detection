#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH -t48:00:00
#SBATCH --mem=40GB
#SBATCH --mail-type=END
#SBATCH --mail-user=zy1129@nyu.edu
#SBATCH --output=slurm_probs_map_%j.out

module purge
module load python3/intel/3.6.3
module load openslide/intel/3.4.1
source ../pyenv/py/3.6.3/bin/activate

python3 wsi/bin/probs_map.py ../data/wsi/$1.tif ckpt/resnet18_crf.ckpt configs/resnet18_crf.json ../data/tissue_mask/$1.npy ../data/probs_map/$1.npy

