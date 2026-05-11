#!/usr/bin/env bash
set -euo pipefail

pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 3
