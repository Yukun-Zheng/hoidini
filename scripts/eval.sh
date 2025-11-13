#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python hoidini/eval/eval.py --device=5

