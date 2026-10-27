#!/bin/bash
# Phase 1: regenerate the London ensembles with exactly 97 capacity units and d_bar as level-2 centre.
cd /home/user/London-Ambulance-Service-System
export PYTHONHASHSEED=0
PY=/home/user/venv/bin/python
OUT=data/derived/real_stations_u97; mkdir -p $OUT logs
for s in 1 2 3 4; do
  $PY -m falcomchain_experiments.las.run_real_stations --seed $s --steps 80000 --snap-every 40 --out-dir $OUT --tag real > logs/u97_s${s}_T80000.log 2>&1 &
done
wait
OUT2=data/derived/real_stations_u97_eps25; mkdir -p $OUT2
for s in 1 2 3 4; do
  $PY -m falcomchain_experiments.las.run_real_stations --seed $s --steps 40000 --snap-every 40 --eps 0.25 --out-dir $OUT2 --tag real > logs/u97_eps25_s${s}_T40000.log 2>&1 &
done
wait
echo "phase1 done $(date)"
