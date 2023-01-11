#!/bin/bash
python train_classification.py run-rose --train --seed 50 --num-epochs 300 --mem-gb 40 \
--timeout-min 420 --force-slurm --slurm-partition "gpu-cluster" \
--concurrency 4 -N 1 -NG 1 --base_logging_frequency 20