#!/bin/bash

if [ -z "\$1" ]; then
    echo "usage: $0 <port> [gpu_id]"
    exit 1
fi

port=$1
gpuid=${2:-0}
export CUDA_VISIBLE_DEVICES="$gpuid"
# export VLLM_ATTENTION_BACKEND=XFORMERS

echo "Port: $port"
echo "GPU ID: $CUDA_VISIBLE_DEVICES"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -m vllm.entrypoints.openai.api_server \
    --model /data/Download/Meta-Llama-3-8B-Instruct \
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
    2>> "${script_dir}/error.log" &

echo $! >> "$(dirname "$0")/launch_llm.pid"

sleep 10

echo "LLM API is running."