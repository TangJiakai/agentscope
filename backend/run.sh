port=9000

if [ $# -eq 1 ]; then
    port=$1
fi

uvicorn backend.app:app --host 0.0.0.0 --port $port --reload