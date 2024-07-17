#!/bin/bash

# get number of server
if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 <server-per-host>"
    exit 1
fi