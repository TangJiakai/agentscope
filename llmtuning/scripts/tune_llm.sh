#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,7 

PARENT_PARENT_DIR="$(realpath "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")"
echo $PARENT_PARENT_DIR

python "$PARENT_PARENT_DIR/code/tune_llm.py"

echo "LLM tuning is done."