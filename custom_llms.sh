#!/bin/bash

uv run --env-file=.env evaluate.py --models=qwen3,llama3,deepseek | tee ./logs/`date +%Y%m%d%H%M%S`.log