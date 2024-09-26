#!/bin/bash

base_port=8083

for gpu_id in {0..7}; do
    port=$(($base_port + gpu_id))
    bash llm/launch_llm.sh $port $gpu_id  &  
done

# 等待所有后台进程完成
wait

echo "所有脚本执行完毕。"