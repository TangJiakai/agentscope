#!/bin/bash

script_dir=$(cd "$(dirname "$0")"; pwd)


# get number of server
if ! [[ "$1" =~ ^[0-9]+$ ]] || ! [[ "$2" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 <server-num-per-host> <base-port>"
    exit 1
fi

server_num_per_host=$1
base_port=$2

> "${script_dir}/.pid"
mkdir -p "${script_dir}/log"

for ((i=0; i<server_num_per_host; i++)); do
    port=$((base_port + i))
    python "${script_dir}/launch_server.py" --base-port ${port} > "${script_dir}/log/${port}.log" 2>&1 &
    echo $! >> "${script_dir}/.pid"
    echo "Started agent server on localhost:${port} with PID $!"
done

echo "All servers started."

python "${script_dir}/assign_host_port.py" --base-port ${base_port} --server-num-per-host ${server_num_per_host}
echo "Assigned base ports to agent configs."
