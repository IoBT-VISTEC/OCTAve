#!/bin/bash
python train_octanetaag.py run-rose --train --seed 50 --coarse-num-epochs 200 --fine-num-epochs 200 --mem-gb 40 \
--timeout-min 420 --force-slurm --slurm-partition "gpu-cluster" \
--concurrency 4 -N 1 -NG 1 --base_logging_frequency 10 --fine-in-channels 3 --monitor val/dice