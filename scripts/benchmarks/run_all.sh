#!/bin/bash

julia -O3 --check-bounds=no --project=../.. benchmark_block_size.jl
julia -O3 --check-bounds=no --project=../.. benchmark_shmem.jl
./weak_scaling.sh