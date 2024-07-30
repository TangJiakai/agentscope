#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,6 
export API_HOST=0.0.0.0 
export API_PORT=8084 
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [ -f "saves/adapter_config.json" ]; then
    llamafactory-cli api configs/llama3_lora.yaml &
else
    llamafactory-cli api configs/llama3.yaml &
fi

echo $! > "$(dirname "$0")/.pid"

sleep 10

echo "LLM API is running."