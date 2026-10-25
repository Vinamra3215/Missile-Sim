#!/usr/bin/env bash
# Reproducible experiment: PPO vs DQN with the same budget (3M steps) for each seed,
# then held-out evaluation and plots.
#
#   bash scripts/run_cpu.sh                       # seeds 1 2, one run at a time (~1 h)
#   PARALLEL=1 SEEDS="1 2 3" bash scripts/run_cpu.sh   # all runs at once (multi-core PC)
set -e
cd "$(dirname "$0")/.."
STEPS=${STEPS:-3e6}
SEEDS=${SEEDS:-"1 2"}
mkdir -p logs

PARALLEL=${PARALLEL:-0}
run() { if [ "$PARALLEL" = 1 ]; then "$@" & else "$@"; fi; }
[ "$PARALLEL" = 1 ] && export OMP_NUM_THREADS=1

for s in $SEEDS; do
  echo "[$(date +%H:%M)] PPO + DQN seed $s  (logs/ppo_s$s.log, logs/dqn_s$s.log)"
  run python train_ppo.py --total-steps $STEPS --seed $s > logs/ppo_s$s.log 2>&1
  run python train_dqn.py --total-steps $STEPS --seed $s > logs/dqn_s$s.log 2>&1
done
wait

echo "[$(date +%H:%M)] evaluating on held-out maps"
python evaluate.py --episodes 300 --levels 0.0 0.5 1.0 1.5 2.0 \
       --policies random steer "models/ppo_s*_best.pth" "models/dqn_s*_best.pth" > logs/eval.log 2>&1
python plot_results.py > logs/plot.log 2>&1
echo "[$(date +%H:%M)] all done"
