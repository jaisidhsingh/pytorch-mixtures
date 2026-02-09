#!/bin/bash
# cd /Users/jaisidhsingh/Code/GitHub/projects/pytorch-mixtures
# eval "$(conda shell.bash hook)"
# conda activate pt
export PYTORCH_ENABLE_MPS_FALLBACK=1
# export PYTHONPATH=./:$PYTHONPATH
python3 -m tests.tests
