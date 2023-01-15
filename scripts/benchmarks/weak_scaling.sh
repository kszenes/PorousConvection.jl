#!/bin/bash -l
nps="1 4 16 25 64"
for np in $nps; do
  srun -t 0:20:0 -N $np -n $np -Cgpu -Aclass04 julia -O3 --check-bounds=no --project=../.. benchmark_distributed.jl
done