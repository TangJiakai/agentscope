#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3 
export API_HOST=0.0.0.0 
export API_PORT=8083
export VLLM_WORKER_MULTIPROC_METHOD=spawn


PARENT_PARENT_DIR="$(realpath "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")"


if [ -f "saves/adapter_config.json" ]; then
    llamafactory-cli api "$PARENT_PARENT_DIR/configs/llama3_lora.yaml" &
else
    llamafactory-cli api "$PARENT_PARENT_DIR/configs/llama3.yaml" &
fi

echo $! > "$(dirname "$0")/.pid"

sleep 10

echo "LLM API is running."