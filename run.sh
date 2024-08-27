#!/bin/bash
if [[ -f .env ]]; then
    set -o allexport && . .env && set +o allexport
fi
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
