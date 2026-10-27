#!/bin/bash
# Phase 5: fixed graph size, growing team count (grid_10000 at w = 10000, 5000, 2500), after phase 4.
cd /home/user/London-Ambulance-Service-System
export PYTHONHASHSEED=0
PY=/home/user/venv/bin/python
while pgrep -f phase34_chain.sh > /dev/null; do sleep 120; done
echo "phase4 finished $(date)"
cd falcomchain_experiments/scalability
for w in 10000 5000 2500; do
  for s in 42 43; do
    timeout 7200 $PY run_scalability.py --sizes 10000 --variant _w$w --steps 3000 --seed $s --out-dir results/fixed_size > ../../logs/fixed_w${w}_s$s.log 2>&1 &
  done
  wait
done
echo "phase5 done $(date)"
