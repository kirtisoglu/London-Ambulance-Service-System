#!/bin/bash
# Phase 3: sensitivity to the demand proxy, uniform within-borough apportionment, four chains of 80,000 steps.
cd /home/user/London-Ambulance-Service-System
export PYTHONHASHSEED=0
PY=/home/user/venv/bin/python
OUT=data/derived/real_stations_u97_uniform; mkdir -p $OUT logs
for s in 1 2 3 4; do
  $PY -m falcomchain_experiments.las.run_real_stations --seed $s --steps 80000 --snap-every 40 --demand-csv data/derived/demand_lsoa_uniform.csv --out-dir $OUT --tag real > logs/u97_uniform_s${s}_T80000.log 2>&1 &
done
wait
echo "phase3 done $(date)"
