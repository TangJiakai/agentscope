#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export VLLM_WORKER_MULTIPROC_METHOD=spawn

PARENT_PARENT_DIR="$(realpath "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")"


if [ -f "saves/adapter_config.json" ]; then
    # llamafactory-cli api "$PARENT_PARENT_DIR/configs/llama3_lora.yaml" &
    python -m vllm.entrypoints.openai.api_server \
        --model /data/pretrain_dir/Meta-Llama-3-8B-Instruct \
        --trust-remote-code \
        --port 8083 \
        --api-key tangjiakai \
        --dtype auto \
        --pipeline-parallel-size 1 \
        --enforce-eager \
        --enable-prefix-caching \
        --enable-lora \
        --lora-modules lora=saves 
else
    # llamafactory-cli api "$PARENT_PARENT_DIR/configs/llama3.yaml" &
    python -m vllm.entrypoints.openai.api_server \
        --model /data/pretrain_dir/Meta-Llama-3-8B-Instruct \
        --trust-remote-code \
        --port 8083 \
        --api-key tangjiakai \
        --dtype auto \
        --pipeline-parallel-size 1 \
        --enforce-eager \
        --enable-prefix-caching
fi

echo $! > "$(dirname "$0")/.pid"

sleep 10

echo "LLM API is running."