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

t=$((SLURM_ARRAY_TASK_ID-1))
p=$((t/5))
s=$((t-p*5))
printf -v p0 "%03d" $p

python3 wsi/bin/tissue_mask.py ../data/wsi_train/patient_$(p0)_node_$(s).tif ../data/tissue_mask_train/patient_$(p0)_node_$(s).npy
