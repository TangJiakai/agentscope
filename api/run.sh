# prompt-dir: the folder path of prompt files
# use-index: use index to replace guided choice

if [ -z "\$1" ]; then
    echo "usage: $0 <port> [gpu_id]"
    exit 1
fi

port=$1
gpuid=${2:-0}
export CUDA_VISIBLE_DEVICES="$gpuid"
# export VLLM_ATTENTION_BACKEND=XFORMERS

echo "Port: $port"
echo "GPU ID: $CUDA_VISIBLE_DEVICES"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

nohup python $script_dir/api_server.py \
  --model /data/pretrain_dir/Meta-Llama-3-8B-Instruct \
  --trust-remote-code \
  --port $port \
  --dtype auto \
  --pipeline-parallel-size 1 \
  --enforce-eager \
  --enable-prefix-caching \
  --disable-frontend-multiprocessing \
  --guided-decoding-backend=lm-format-enforcer \
  --gpu-memory-utilization 0.8 \
  --prompt-dir /data/tangjiakai/general_simulation/simulation/examples/job_seeking/prompts \
  --use-index \
  2> "$script_dir/api_server_$port.log" &

echo $! >> "$(dirname "$0")/launch_llm.pid"


