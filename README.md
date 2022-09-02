# Replication package: Noarr Layout Traversal

This is a replication package containing code and measurements scripts related to a GCO paper titled:  **A Pure C++ Approach to Optimal Traversal of Data Structures**

## Measurements Overview

This package contains scripts that perform following measurements:

* Matrix multiplication
* Stencil computation
* Matrix copy
* CUDA Hadamard product

## Requirements for running the measurements

Hardware requirements:

* To replicate all tests, a CUDA-compatible GPU is required.

Software requirements:

* `g++` (version 12)
* `clang++` (version 15)
* [CUDA toolkit 11.7 or later](https://developer.nvidia.com/cuda-downloads) and appropriate driver (for GPU tests)


## Running the measurements

To execute all measurements, run:
```
./run_all.sh
```
Without a CUDA-compatible GPU, you can run:
```
./run_cpu_only.sh
```
which will skip CUDA Hadamard product measurement.

When the script finishes, multiple `txt` files will be created in `data` folder. The name of each file is a composition of _a compiler that built the executable_ and a _measurement name_, such as `g++_hadamard.txt`.

