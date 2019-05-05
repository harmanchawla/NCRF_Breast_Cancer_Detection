#!/bin/bash

module purge
module load python3/intel/3.6.3
module load openslide/intel/3.4.1
source ../pyenv/py/3.6.3/bin/activate

for file in $(ls ../data/wsi/); do
    filename=$(file%.*)
    python3 wsi/bin/tissue_mask.py ../data/wsi/$filename.tif ../data/tissue_mask/$filename.npy
done

