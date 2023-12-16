#!/bin/bash

: ${NODES:=1}

salloc -N $NODES --exclusive --partition=class1  --gres=gpu:4     \
  mpirun --bind-to none -mca btl ^openib -npernode 1              \
  numactl --physcpubind 0-63                                      \
  ./classifier $@
