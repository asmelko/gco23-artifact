#!/bin/bash

mkdir data

function run_with_compiler()
{
    nvcc -O3 -ccbin $1 -std=c++17 --expt-relaxed-constexpr -allow-unsupported-compiler main.cu -o run_traverser_cuda
    $1 -std=c++17 -O3 main.cpp -o run_traverser_cpu

    ./run_traverser_cuda 3200 3200 > data/$1_hadamard.txt
    ./run_traverser_cpu matmul 1600 1600 1600 > data/$1_matmul.txt
    ./run_traverser_cpu stencil 960 960 960 > data/$1_stencil.txt
    ./run_traverser_cpu transpose 8192 8192 > data/$1_transpose.txt
}

run_with_compiler g++
run_with_compiler clang++