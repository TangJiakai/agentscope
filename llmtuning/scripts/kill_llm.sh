#!/bin/bash

PID_FILE="$(dirname "$0")/.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    kill $PID
    echo "Process $PID has been terminated."
    rm "$PID_FILE"  
else
    echo "No PID file found."
fi