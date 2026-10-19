#!/bin/sh
cd "$(dirname "$0")" || exit 1
./.venv/bin/python app.py &
SERVER_PID=$!
sleep 1
open http://127.0.0.1:8756
wait $SERVER_PID
