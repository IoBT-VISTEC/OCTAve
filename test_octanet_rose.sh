#!/bin/bash
python test_octanetagg.py run-rose --train --seed 50 --coarse-num-epochs 300 --fine-num-epochs 100 --ran-vanilla --mem-gb 40 \
--timeout-min 420 --force-slurm --slurm-partition "gpu-cluster" \
--concurrency 4 -N 1 -NG 1 --base_logging_frequency 100