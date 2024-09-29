#!/bin/bash

port_list=(8084 8085)
gpu_list=(6 7)

script_path=$(cd `dirname $0`; pwd)

for i in "${!port_list[@]}"; do
    port=${port_list[$i]}
    gpu_id=${gpu_list[$i]}
    bash $script_dir/launch_llm.sh $port $gpu_id &
done

wait

echo "All LLM API servers are running."