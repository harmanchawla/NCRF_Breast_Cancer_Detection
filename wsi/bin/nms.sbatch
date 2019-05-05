#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH -t48:00:00
#SBATCH --mem=40GB
#SBATCH --mail-type=END
#SBATCH --mail-user=zy1129@nyu.edu
#SBATCH --output=slurm_tumor_coord_%j.out

module purge
module load python3/intel/3.6.3
module load openslide/intel/3.4.1
source ../pyenv/py/3.6.3/bin/activate

for file in $(ls ../data/probs_map/); do
    filename=${file%.*}
    python3 wsi/bin/nms.py ../data/probs_map/$filename.npy ../data/tumor_coord/$filename.csv
done


