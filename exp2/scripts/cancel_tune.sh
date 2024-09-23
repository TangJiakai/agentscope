#!/bin/bash

PID_FILE="$(realpath "$(dirname "${BASH_SOURCE[0]}")")/tune_llm.pid"
PARENT_PARENT_DIR="$(realpath "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")"

if [ -f "$PID_FILE" ]; then
    while IFS= read -r PID; do
        if kill -0 "$PID" 2>/dev/null; then 
            kill -9 "$PID"
            echo "Process $PID has been terminated."
        else
            echo "Process $PID does not exist."
        fi
    done < "$PID_FILE"
    rm "$PID_FILE"
else
    echo "No PID file found."
fi

rm -rf "$PARENT_PARENT_DIR/tmp_saves

echo "Tuning process has been terminated."