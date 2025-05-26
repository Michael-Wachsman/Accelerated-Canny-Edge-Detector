# Accelerated Canny Edge Detection

This project implements an optimized Canny edge detection algorithm<sup>[1]</sup> with significant performance improvements through serial optimizations, OpenMP parallelization, and CUDA GPU acceleration.

## Overview



![Edge Detection Example](/images/image.png)

The Canny edge detector (shown above) is a multi-stage algorithm known for its accuracy and noise suppression, making it ideal for applications like object recognition and medical imaging. In contrast, the Sobel edge detector is a faster, gradient-based method suited for real-time systems or quick preprocessing tasks where precision is less critical. Despite Canny’s superior edge detection capabilities, its computational complexity poses challenges for real-time applications and high-resolution images. Compared to a basic serial implementation of the Canny Edge detector, this project optimizes the Canny Edge Detection algorithm to reduce execution time on large images by >1000x.*

<sub>*: Since the Canny edge detector wraps the Sobel edge detector, it is worth noting that our implementation of the Sobel edge detector reduces execution time by >5000x compared to a basic implementation</sub>

## Features

- Full Canny Edge Detection implementation (Gaussian smoothing, Sobel filter, gradient thresholding, hysteresis)
- Multiple optimization techniques:
  - Serial optimizations: blocking, separable filters, compiler flags
  - Parallel CPU implementation using OpenMP
  - GPU acceleration using CUDA
- Comprehensive performance analysis plot generation

## Performance

All benchmarks were obtained on NERSC's Perlmutter supercomputer, runtimes may vary depending on hardware.

| Implementation                    | Runtime (seconds) | Speedup factor |
|----------------------------------|-------------------|----------------|
| Basic Serial (Initial Canny)     | 23.71             | 1.0x           |
| Serial with Blocking             | ~10.0             | ~2.4x          |
| Serial with Separable Filters    | ~5.0              | ~4.7x          |
| Serial + Compiler Flags          | ~1.0              | ~23.7x         |
| OpenMP (32 threads)              | ~0.05              | ~474.2x          |
| Basic GPU                        | ~0.1              | ~237.1x          |
| GPU with Separable Filters       | ~0.02             | ~1185.5x         |

All measurements taken on a 3744 x 5616 pixel test image running on NERSC's Perlmutter supercomputer.

## Requirements

- C++ compiler with C++11 support
- OpenMP for parallel CPU implementation
- CUDA toolkit (11.0+) for GPU implementations
- OpenCV for image I/O
- Python with NumPy, Matplotlib for verification scripts

## Usage

```bash
# Basic usage
./canny_edge --input <input_image> --output <output_image> --implementation <impl_type>

# Implementation types:
# - basic_serial: Initial Canny implementation
# - block_serial: Serial implementation with blocking
# - separable_serial: Serial with separable filters
# - omp_parallel: OpenMP parallelized implementation
# - basic_gpu: Basic CUDA implementation
# - separable_gpu: CUDA with separable filters and shared memory

# Example
./canny_edge --input sample.jpg --output edges.png --implementation separable_gpu
```

## Implementation Details

### Serial Optimizations

1. The image is divided into smaller blocks to improve cache utilization and spatial locality.
2. A seperable Gaussian filter is used reducing filtering complexity from O(N²) to O(N) w/r/t filter size.

### OpenMP Parallelization

- Employs thread-level parallelism for all major computation loops
- Shows near-linear scaling up to 16-32 threads

### CUDA GPU Implementation
   - Optimized memory access patterns for memory coalescing
   - Shared memory utilization for faster access
   - Block-based processing with 32×32 block size
   - GPU-based hysteresis implementation
   - Optimized arctangent to remove the need to utilize the GPU's SFU

## Results and Analysis

### Performance Scaling

![Performance Scaling](/plots/loglog_7_comparison.png)

The log-log plot shows linear scaling with image size for all implementations, with the GPU implementations showing the lowest slope.

## Verification

Correctness was verified by comparing outputs between implementations:
- Block Serial: 0% difference from baseline
- Separable Serial: 0.168% difference
- OpenMP: 0.227% difference
- Basic GPU: 0.002% difference
- Separable GPU: 0.97% difference (due to unavoidable hysteresis algorithm race conditions)

## Contributors

- Qishen Peng
- Cody Zheng 
- Dengyu Tu
- Yiwei Luo

## Acknowledgments
- Thank you to the Cornell Parllel Processing and High Performance Computing class for supplying our team with the necessary NERSC credits 

## Refrences 
[1] Canny edge detector. (n.d.). In Wikipedia. Retrieved February 25, 2025, from https://en.wikipedia.org/wiki/Canny_edge_detector


