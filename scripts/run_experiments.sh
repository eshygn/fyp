#!/bin/bash
# =============================================================
# Master experiment script for Dissertation
# Run each section one at a time inside a tmux session
# =============================================================

# ----- STEP 0: Setup (run once) -----
# cd ~/dissertation/scripts
# source ~/dissertation_env/bin/activate

# ----- STEP 1: Prepare data -----
echo "=== Step 1: Preparing dataset ==="
python prepare_data.py \
    --input ../data/filtered_Qwen3-4B.jsonl \
    --output_dir ../data/prepared \
    --n_train 2500 \
    --n_test 100 \
    --seed 42

# ----- STEP 2: Test run (SMALL - checks everything works) -----
# Uncomment this to do a quick sanity check with 50 samples, 20 steps
# python train_dpo.py \
#     --mode dpo \
#     --beta 0.1 \
#     --run_name test_run \
#     --max_steps 20 \
#     --batch_size 1 \
#     --logging_steps 5

# ----- STEP 3: Standard DPO - Beta ablation -----
echo "=== Step 3a: Standard DPO, beta=0.1 ==="
python train_dpo.py --mode dpo --beta 0.1 --run_name dpo_b01

echo "=== Step 3b: Standard DPO, beta=0.3 ==="
python train_dpo.py --mode dpo --beta 0.3 --run_name dpo_b03

echo "=== Step 3c: Standard DPO, beta=0.5 ==="
python train_dpo.py --mode dpo --beta 0.5 --run_name dpo_b05

# ----- STEP 4: Length-normalised DPO - Beta ablation -----
echo "=== Step 4a: LN-DPO, beta=0.1 ==="
python train_dpo.py --mode dpo_ln --beta 0.1 --run_name dpo_ln_b01

echo "=== Step 4b: LN-DPO, beta=0.3 ==="
python train_dpo.py --mode dpo_ln --beta 0.3 --run_name dpo_ln_b03

echo "=== Step 4c: LN-DPO, beta=0.5 ==="
python train_dpo.py --mode dpo_ln --beta 0.5 --run_name dpo_ln_b05

# ----- STEP 5: Generate stories from each condition -----
echo "=== Step 5: Generating stories ==="

# Baseline (no adapter)
python generate.py --output ../results/baseline_stories.jsonl

# Best standard DPO (pick best beta after reviewing training logs)
python generate.py --adapter ../models/dpo_b01/final --output ../results/dpo_b01_stories.jsonl

# Best LN-DPO
python generate.py --adapter ../models/dpo_ln_b01/final --output ../results/dpo_ln_b01_stories.jsonl

echo "=== All experiments complete ==="
