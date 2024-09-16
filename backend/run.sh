#!/bin/bash

export HOST=0.0.0.0
export PORT=9111

if [ $# -eq 1 ]; then
    export PORT=$1
fi

uvicorn backend.app:app --host $HOST --port $PORT