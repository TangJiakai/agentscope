#!/bin/bash

base_port=8666

script_path=$(cd `dirname $0`; pwd)

for gpu_id in {0..7}; do
    port=$(($base_port + gpu_id))
    python $script_path/launch_emb_model.py --gpu $gpu_id --port $port &  
done

sleep 10
echo "所有脚本执行完毕。"