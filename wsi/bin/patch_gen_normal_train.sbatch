#!/bin/bash

#SBATCH --node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH -t48:00:00
#SBATCH --mem=40GB
#SBATCH --mail-type=END
#SBATCH --mail-user=zy1129@nyu.edu
#SBATCH --output=patch_gen_normal_train_%j.out

module purge
module load python3/intel/3.6.3
module load openslide/intel/3.4.1
source ../pyenv/py3.6.3/bin/activate

python3 wsi/bin/patch_gen.py ../data16/wsi_train coords/normal_train_sample.txt ../data16/patches_normal_train/

