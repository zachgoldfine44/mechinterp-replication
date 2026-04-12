#!/bin/bash
# Run critique followup experiments on all 5 remaining models
# (Qwen-7B already done in previous session)

set -e

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN environment variable before running}"
export TORCHDYNAMO_DISABLE=1

cd /root/mechinterp-replication

echo "========================================="
echo "Starting critique followups on 5 models"
echo "========================================="

# Medium tier first (most valuable data)
echo ""
echo "=== Model 1/5: Llama-3.1-8B-Instruct ==="
python3 scripts/critique_followups_hf.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --model-key llama_8b

echo ""
echo "=== Model 2/5: Gemma-2-9B-IT ==="
python3 scripts/critique_followups_hf.py \
    --model google/gemma-2-9b-it \
    --model-key gemma_9b

# Small tier (pipeline validation + scaling floor)
echo ""
echo "=== Model 3/5: Llama-3.2-1B-Instruct ==="
python3 scripts/critique_followups_hf.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --model-key llama_1b

echo ""
echo "=== Model 4/5: Qwen-2.5-1.5B-Instruct ==="
python3 scripts/critique_followups_hf.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --model-key qwen_1_5b

echo ""
echo "=== Model 5/5: Gemma-2-2B-IT ==="
python3 scripts/critique_followups_hf.py \
    --model google/gemma-2-2b-it \
    --model-key gemma_2b

echo ""
echo "========================================="
echo "ALL 5 MODELS COMPLETE"
echo "========================================="
