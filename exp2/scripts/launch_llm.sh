#!/bin/bash

if [ -z "\$1" ]; then
    echo "usage: $0 <port> [gpu_id]"
    exit 1
fi

port=$1
gpuid=${2:-0}
export CUDA_VISIBLE_DEVICES="$gpuid"
export VLLM_ATTENTION_BACKEND=XFORMERS

echo "GPU ID: $CUDA_VISIBLE_DEVICES"
echo "Port: $port"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_PARENT_DIR="$(realpath "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")"

if [ -f "${PARENT_PARENT_DIR}/exp2/saves/adapter_config.json" ]; then
    python -m vllm.entrypoints.openai.api_server \
        --model /data/pretrain_dir/Meta-Llama-3-8B-Instruct \
        --trust-remote-code \
        --port $port \
        --dtype auto \
        --pipeline-parallel-size 1 \
        --enforce-eager \
        --enable-prefix-caching \
        --enable-lora \
        --lora-modules lora="${script_dir}/exp2/saves" \
        --disable-frontend-multiprocessing \
        --guided-decoding-backend=lm-format-enforcer \
        --gpu-memory-utilization 0.8 \
        >> "${script_dir}/llm.log" 2>> "${script_dir}/.log" &
else
    python -m vllm.entrypoints.openai.api_server \
        --model /data/pretrain_dir/Meta-Llama-3-8B-Instruct \
        --trust-remote-code \
        --port $port \
        --dtype auto \
        --pipeline-parallel-size 1 \
        --enforce-eager \
        --enable-prefix-caching \
        --enable-lora \
        --disable-frontend-multiprocessing \
        --guided-decoding-backend=lm-format-enforcer \
        --gpu-memory-utilization 0.9 \
        >> "${script_dir}/llm.log" >> "${script_dir}/.log" &
fi

echo $! >> "$(dirname "$0")/.pid"

sleep 10

echo "LLM API is running."