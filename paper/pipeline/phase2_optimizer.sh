#!/bin/bash
# Phase 2: FalCom as an optimizer on the London instance (Section 7.3 rule), eight seeds, T = 50,000.
cd /home/user/London-Ambulance-Service-System
export PYTHONHASHSEED=0
PY=/home/user/venv/bin/python
OUT=data/derived/real_stations_u97_opt; mkdir -p $OUT logs
for batch in "11 12 13 14" "15 16 17 18"; do
  for s in $batch; do
    $PY -m falcomchain_experiments.las.run_real_stations --seed $s --steps 50000 --snap-every 5000 --optimizer --out-dir $OUT --tag opt > logs/u97_opt_s${s}_T50000.log 2>&1 &
  done
  wait
done
echo "phase2 done $(date)"
