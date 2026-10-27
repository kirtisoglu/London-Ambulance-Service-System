#!/bin/bash
# Phases 3, 4 and 5 and the long 4x6 enumeration (PID-based waits only).
cd /home/user/London-Ambulance-Service-System
export PYTHONHASHSEED=0
PY=/home/user/venv/bin/python
echo "phase2 finished $(date)"
bash /home/user/scratch/phase3_uniform.sh > logs/phase3.log 2>&1 &
P3=$!
$PY -m falcomchain_experiments.validation.enumeration --instance 4x6 --steps 3000000 --out-dir data/derived/validation_v2 > logs/enumeration_4x6_long.log 2>&1 &
P46=$!
wait $P3
echo "phase3 done $(date)"
cd falcomchain_experiments/scalability
for s in 42 43 44 45; do
  $PY run_scalability.py --sizes 50000 --steps 30000 --seed $s --track-structural --snap-every 60 --out-dir results/multistart > ../../logs/ms50000_s$s.log 2>&1 &
done
wait
echo "multistart 50000 done $(date)"
for s in 42 43 44 45; do
  $PY run_scalability.py --sizes 10000 --steps 30000 --seed $s --track-structural --snap-every 60 --out-dir results/multistart > ../../logs/ms10000_s$s.log 2>&1 &
done
$PY run_scalability.py --sizes 100 400 1000 10000 50000 --steps 10000 --seed 42 > ../../logs/sweep.log 2>&1 &
wait
echo "multistart 10000 and sweep done $(date)"
$PY analyze_multistart.py --nodes 10000 --seeds 42 43 44 45 > ../../logs/analyze_ms10000.log 2>&1
$PY analyze_multistart.py --nodes 50000 --seeds 42 43 44 45 > ../../logs/analyze_ms50000.log 2>&1
$PY analyze_scalability.py > ../../logs/analyze_scal.log 2>&1
for w in 10000 5000 2500; do
  for s in 42 43; do
    timeout 7200 $PY run_scalability.py --sizes 10000 --variant _w$w --steps 3000 --seed $s --out-dir results/fixed_size > ../../logs/fixed_w${w}_s$s.log 2>&1 &
  done
  wait
done
echo "phase5 done $(date)"
cd ../..
$PY -m falcomchain_experiments.gurobi.run_falcom_multistart 400_dense --steps 10000 > logs/dense400_opt.log 2>&1
echo "phase4 done $(date)"
