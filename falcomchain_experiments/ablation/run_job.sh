#!/bin/bash
# one line of logs/jobs.txt: <grid|london> <size> <variant> <steps> <seed>; 60-minute cap per run
set -u
kind=$1; size=$2; variant=$3; steps=$4; seed=$5
cd "$(dirname "$0")/../.."
export PYTHONHASHSEED=0
log=falcomchain_experiments/ablation/logs/${kind}_${size}_${variant}_s${seed}.log
if [ "$kind" = grid ]; then
  timeout 40m python3 -m falcomchain_experiments.ablation.run_grid --size $size --variant $variant --steps $steps --seed $seed > $log 2>&1
else
  timeout 40m python3 -m falcomchain_experiments.ablation.run_london --variant $variant --steps $steps --seed $seed > $log 2>&1
fi
echo "exit $?" >> $log
