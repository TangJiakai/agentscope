#!/bin/bash

script_dir=$(cd "$(dirname "$0")"; pwd)

# get number of server
if ! [[ "$1" =~ ^[0-9]+$ ]] || ! [[ "$2" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 <server_num_per_host> <base_port>"
    exit 1
fi

server_num_per_host=$1
base_port=$2

> "${script_dir}/.pid"

for ((i=0; i<server_num_per_host; i++)); do
    port=$((base_port + i))
    python "${script_dir}/launch_server.py" --base_port ${port} &
    echo $! >> "${script_dir}/.pid"
    echo "Started agent server on localhost:${port} with PID $!"
done

echo "All servers started."

python "${script_dir}/assign_host_port.py" --base_port ${base_port} --server_num_per_host ${server_num_per_host}
echo "Assigned base ports to agent configs."
