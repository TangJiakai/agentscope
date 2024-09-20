#!/bin/bash

if [ -z "\$1" ]; then
    echo "usage: $0 <port> [gpu_id]"
    exit 1
fi

port=$1
gpuid=${2:-0}
export CUDA_VISIBLE_DEVICES="$gpuid"

echo "GPU ID: $CUDA_VISIBLE_DEVICES"
echo "Port: $port"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "llmtuning/saves/adapter_config.json" ]; then
    python -m vllm.entrypoints.openai.api_server \
        --model your_llm_model_path \
        --trust-remote-code \
        --port $port \
        --dtype auto \
        --pipeline-parallel-size 1 \
        --enforce-eager \
        --enable-prefix-caching \
        --enable-lora \
        --lora-modules lora=llmtuning/saves \
        --disable-frontend-multiprocessing \
        --guided-decoding-backend=lm-format-enforcer \
        2>> "${script_dir}/error.log" &
else
    python -m vllm.entrypoints.openai.api_server \
        --model your_llm_model_path \
        --trust-remote-code \
        --port $port \
        --dtype auto \
        --pipeline-parallel-size 1 \
        --enforce-eager \
        --enable-prefix-caching \
        --enable-lora \
        --disable-frontend-multiprocessing \
        --guided-decoding-backend=lm-format-enforcer \
        2>> "${script_dir}/error.log" &
fi

echo $! >> "$(dirname "$0")/.pid"

sleep 10

echo "LLM API is running."