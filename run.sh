#!/bin/bash

uv run --env-file=.env evaluate.py | tee ./logs/`date +%Y%m%d%H%M%S`.log