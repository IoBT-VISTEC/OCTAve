#!/bin/bash
python train_octanet.py run-octa500 --train --seed 50 --folds 5 --num-epochs 1000 --mem-gb 40 \
--timeout-min 420 --force-slurm --slurm-partition "gpu-cluster" --exclude-node ist-gpu-[01,02,04,05,06,07,15] \
--concurrency 4 -N 1 -NG 1 --base_logging_frequency 100 --scribble-presence-ratio 1.0