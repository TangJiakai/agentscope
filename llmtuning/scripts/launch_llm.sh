#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
# export VLLM_WORKER_MULTIPROC_METHOD=spawn

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


if [ -f "llmtuning/saves/adapter_config.json" ]; then
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
        --lora-modules lora=llmtuning/saves \
        2>> "${script_dir}/error.log" &
else
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
        2>> "${script_dir}/error.log" &
fi

echo $! > "$(dirname "$0")/.pid"

sleep 10

echo "LLM API is running."