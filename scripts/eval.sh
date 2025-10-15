#!/bin/bash
conda activate hoidini
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python hoidini/eval/eval.py

