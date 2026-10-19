#!/bin/bash
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install llvm-dev
git clone --depth 1 --branch v2.8.3 https://github.com/Dao-AILab/flash-attention.git third_party/flash-attention
git -C third_party/flash-attention submodule update --init --depth 1 csrc/cutlass

