#!/bin/bash

port_list=(8084 8085)
gpu_list=(6 7)

script_path=$(cd `dirname $0`; pwd)

for i in "${!port_list[@]}"; do
    port=${port_list[$i]}
    gpu_id=${gpu_list[$i]}
    bash $script_dir/launch_emb_model.sh --gpu $gpu_id --port $port &
done

sleep 10
echo "All embedding models are running."