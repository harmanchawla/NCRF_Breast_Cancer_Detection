#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH -t48:00:00
#SBATCH --mem=40GB
#SBATCH --mail-type=END
#SBATCH --mail-user=zy1129@nyu.edu

module purge
module load python3/intel/3.6.3
source ../pyenv/py3.6.3/bin/activate

python3 wsi/bin/train.py configs/unet_crf.json saved_model/
