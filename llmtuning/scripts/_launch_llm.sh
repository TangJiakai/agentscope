#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "错误: 需要两个参数"
    echo "用法: \$0 <port> <gpuid>"
    exit 1
fi

port=$1
gpuid=$2
export CUDA_VISIBLE_DEVICES="$gpuid"

echo "GPU ID: $CUDA_VISIBLE_DEVICES"
echo "Port: $port"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_dir="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")" && pwd)"


python "$python_dir/code/launch_llm.py" --port $port 2> "${script_dir}/llm_error.log" | tee "${script_dir}/llm.log" &

# 获取后台启动的 Python 进程 PID
py_pid=$!

# 等待片刻，确保 Python 进程已经启动
sleep 1

# 验证该进程号是否属于 Python 进程并且匹配相应的脚本
if ps -p $py_pid -o args= | grep -q "python $python_dir/code/launch_llm.py"; then
    real_pid=$py_pid
else
    # 如果无法确认进程，尝试通过 pgrep 搜索精确匹配的 Python 进程
    real_pid=$(pgrep -f "python $python_dir/code/launch_llm.py")
fi

# 将确定的 PID 写入 .pid 文件
echo $real_pid >> "$(dirname "$0")/.pid"

sleep 5
echo "LLM API is running."