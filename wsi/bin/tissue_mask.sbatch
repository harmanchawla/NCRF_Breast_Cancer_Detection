#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH -t48:00:00
#SBATCH --mem=40GB
#SBATCH --mail-type=END
#SBATCH --mail-user=zy1129@nyu.edu
#SBATCH --output=slurm_tissue_mask_%j.out

module purge
module load python3/intel/3.6.3
module load openslide/intel/3.4.1
source ../pyenv/py3.6.3/bin/activate

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PATIENT=$((t/5))
SLIDE_ID=$((t-p*5))
printf -v PATIENT_ID "%03d" $PATIENT

echo 'python3 wsi/bin/tissue_mask.py ../data/wsi_train/patient_$PATIENT_ID_node_$SLIDE_ID.tif ../data/tissue_mask_train/patient_$PATIENT_ID_node_$SLIDE_ID.npy'

python3 wsi/bin/tissue_mask.py ../data/wsi_train/patient_$PATIENT_ID_node_$SLIDE_ID.tif ../data/tissue_mask_train/patient_$PATIENT_ID_node_$SLIDE_ID.npy
