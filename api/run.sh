# prompt-dir: the folder path of prompt files
# use-index: use index to replace guided choice

CUDA_VISIBLE_DEVICES=1 nohup python api_server.py \
  --model /data/pretrain_dir/Meta-Llama-3-8B-Instruct \
  --trust-remote-code \
  --port  9120\
  --dtype auto \
  --pipeline-parallel-size 1 \
  --enforce-eager \
  --enable-prefix-caching \
  --enable-lora \
  --lora-modules lora=llmtuning/saves \
  --disable-frontend-multiprocessing \
  --guided-decoding-backend=lm-format-enforcer \
  --prompt-dir  /data/wl/new/agentscope/simulation/examples/job_seeking/prompts \
  --use-index > api_server_index_9120.log 2>&1 &

