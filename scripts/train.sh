#!/usr/bin/env bash

# The -x option is one of the options of the set command that is used to enable the debugging feature of the bash script for troubleshooting.
set -x
GPUS=$1  # 将命令行的第一个参数赋值给变量 GPUS

PY_ARGS=${@:2} # 将命令行的第二个及后面的参数赋值给变量 PY_ARGS

CUDA_VISIBLE_DEVICES=${GPUS} python main.py ${PY_ARGS}
