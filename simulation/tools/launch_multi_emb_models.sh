#!/bin/bash

port_list=(8083 8084 8085 8086 8087 8088 8089 8090)
gpu_list=(0 1 2 3 4 5 6 7)

script_dir=$(cd `dirname $0`; pwd)

for i in "${!port_list[@]}"; do
    port=${port_list[$i]}
    gpu_id=${gpu_list[$i]}
    python "$script_dir/launch_emb_model.py" --gpu $gpu_id --port $port &
done

sleep 10
echo "All embedding models are running."