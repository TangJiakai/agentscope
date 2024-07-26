#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

export PATH="$PARENT_DIR:$PATH"

echo "Updated PATH: $PARENT_DIR"