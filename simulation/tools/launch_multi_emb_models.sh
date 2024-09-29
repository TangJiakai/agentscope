#!/bin/bash

port_list=(8666 8667 8668 8669 8670 8671 8672 8673)
gpu_list=(0 1 2 3 4 5 6 7)

script_path=$(cd `dirname $0`; pwd)
pid_file="$script_path/.pid"

# 如果 .pid 文件不存在，则创建一个新文件
if [[ ! -f $pid_file ]]; then
    touch $pid_file
fi

script_dir=$(cd `dirname $0`; pwd)

for i in "${!port_list[@]}"; do
    port=${port_list[$i]}
    gpu_id=${gpu_list[$i]}
    python "$script_dir/launch_emb_model.py" --gpu $gpu_id --port $port &
    echo $! >> $pid_file 
done

sleep 10
echo "All embedding models are running."