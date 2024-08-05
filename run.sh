#!/bin/bash
if [[ -f .env ]]; then
    set -o allexport && . .env && set +o allexport
fi
uvicorn main:app
