#!/bin/bash
# Run sycophancy v2 with ChatGPT judge across all 6 models
set -e

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
export TORCHDYNAMO_DISABLE=1

cd /root/mechinterp-replication

echo "========================================="
echo "Sycophancy v2 with ChatGPT judge - all 6 models"
echo "========================================="

# Medium tier first (most valuable)
echo ""
echo "=== Model 1/6: Qwen-2.5-7B-Instruct ==="
python3 scripts/sycophancy_v2.py \
    --model Qwen/Qwen2.5-7B-Instruct --model-key qwen_7b --judge chatgpt

echo ""
echo "=== Model 2/6: Llama-3.1-8B-Instruct ==="
python3 scripts/sycophancy_v2.py \
    --model meta-llama/Llama-3.1-8B-Instruct --model-key llama_8b --judge chatgpt

echo ""
echo "=== Model 3/6: Gemma-2-9B-IT ==="
python3 scripts/sycophancy_v2.py \
    --model google/gemma-2-9b-it --model-key gemma_9b --judge chatgpt

# Small tier
echo ""
echo "=== Model 4/6: Qwen-2.5-1.5B-Instruct ==="
python3 scripts/sycophancy_v2.py \
    --model Qwen/Qwen2.5-1.5B-Instruct --model-key qwen_1_5b --judge chatgpt

echo ""
echo "=== Model 5/6: Llama-3.2-1B-Instruct ==="
python3 scripts/sycophancy_v2.py \
    --model meta-llama/Llama-3.2-1B-Instruct --model-key llama_1b --judge chatgpt

echo ""
echo "=== Model 6/6: Gemma-2-2B-IT ==="
python3 scripts/sycophancy_v2.py \
    --model google/gemma-2-2b-it --model-key gemma_2b --judge chatgpt

echo ""
echo "========================================="
echo "ALL 6 MODELS COMPLETE"
echo "========================================="
