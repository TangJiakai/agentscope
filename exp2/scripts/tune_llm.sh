#!/bin/bash

export CUDA_VISIBLE_DEVICES="4,5"

PARENT_PARENT_DIR="$(realpath "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <tuning-mode>"
    exit 1
fi

python "$PARENT_PARENT_DIR/code/tune_llm.py" --tuning_mode $1 & 

echo $! >> "$(dirname "$0")/tune_llm.pid"

echo "LLM tuning is done."