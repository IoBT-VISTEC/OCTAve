#!/bin/bash
python train_scribble_rose_fully.py run-octa500 --train --seed 50 --num-epochs 300 \
--mem-gb 40 --timeout-min 480 --force-slurm --slurm-partition gpu-cluster --concurrency 4 -N 1 -NG 1 \
--base_logging_frequency 20 --scribble-presence-ratio 1.0 --experiment-start-index 0 \
