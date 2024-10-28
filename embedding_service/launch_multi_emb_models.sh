#!/bin/bash

export MKL_THREADING_LAYER="GNU"

port_list=(8666)
gpu_list=(0)
model_path="/data/pretrain_dir/m3e-base"

current_dir=$(cd `dirname $0`; pwd)
PID_FILE="${current_dir}/.pid"
LOG_FILE="${current_dir}/.log"

script_dir=$(cd `dirname $0`; pwd)

for i in "${!port_list[@]}"; do
    port=${port_list[$i]}
    gpu_id=${gpu_list[$i]}
    python "$script_dir/launch_emb_model.py" \
        --gpu $gpu_id \
        --port $port \
        --model_path $model_path \
        >> $LOG_FILE 2>&1 &
    echo $! >> $PID_FILE 
done

sleep 10
echo "All embedding models are running."