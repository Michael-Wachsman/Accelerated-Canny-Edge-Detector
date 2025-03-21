#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/types.h>

#include "../common/solver.hpp"

#define BLOCK_SIZE 32

#define KERNEL_SIZE 7
#define SIGMA 1.4

#define FILTER_SIZE 5
#define HALF_FILTER_SIZE FILTER_SIZE / 2

#define NORMAL 1.0f / (2.0f * M_PI * SIGMA * SIGMA)
#define LOW_THRESH_RATIO .05
#define HIGH_THRESH_RATIO .2
#define WEAK 25
#define STRONG 255

int size, width, height, blockNum;

uint8_t *input_image;
float *gray_image;
float *smooth_image;
float *direction_matrix;
float *gpu_kernel;
float *gpu_gx;
float *gpu_gy;

void free() {
  cudaFree(input_image);
  cudaFree(gray_image);
  cudaFree(smooth_image);
  cudaFree(direction_matrix);
  cudaFree(gpu_kernel);
  cudaFree(gpu_gx);
  cudaFree(gpu_gy);
}

void generate_gaussian_kernel(float *kernel);
__global__ void smooth_gaussian(float *input, float *output, float *gpu_kernel,
                                int width, int height, int size);

void generate_sobel_filter(float *filter, bool is_x_direction);
__global__ void sobel(float *smooth_image, float *gradient_matrix,
                      float *direction_matrix, float *gpu_gx, float *gpu_gy,
                      int width, int height, int size);
__global__ void grad_mag_thresh(float *gradient_matrix, float *direction_matrix,
                                int width, int height, int size);

__global__ void max_kernel(const float *image, float *reduced_max, int size);
__global__ void double_thresh_kernel(float *image, int width, int height,
                                     int size, float high_thresh,
                                     float low_thresh);
void double_thresh(float *image);

__global__ void cast_int(float *input, uint8_t *output, int width, int height,
                         int size);
void hysteresis(uint8_t *output);

__global__ void gpu_grayscale_magic(uint8_t *image, float *out_image, int width,
                                    int height, int size, int num_color);

void findEdges(uint8_t *image, uint8_t *out_image, int w, int h,
               int num_color) {
  width = w;
  height = h;
  size = width * height;
  // dim3 blockNum(BLOCK_SIZE, BLOCK_SIZE);
  float kernel[KERNEL_SIZE];
  float gx[FILTER_SIZE * FILTER_SIZE];
  float gy[FILTER_SIZE * FILTER_SIZE];

  // Step 0 - Generate Kernels
  cudaEvent_t start, stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  generate_gaussian_kernel(kernel);
  generate_sobel_filter(gx, true);
  generate_sobel_filter(gy, false);

  cudaMalloc(&input_image, num_color * size * sizeof(uint8_t));
  cudaMalloc(&gray_image, size * sizeof(float));
  cudaMalloc(&smooth_image, size * sizeof(float));
  cudaMalloc(&direction_matrix, size * sizeof(float));
  cudaMalloc(&gpu_kernel, KERNEL_SIZE * sizeof(float));
  cudaMalloc(&gpu_gx, FILTER_SIZE * FILTER_SIZE * sizeof(float));
  cudaMalloc(&gpu_gy, FILTER_SIZE * FILTER_SIZE * sizeof(float));

  // Asynchronous memory transfers
  cudaMemcpyAsync(input_image, image, num_color * size * sizeof(uint8_t),
                  cudaMemcpyHostToDevice, 0);
  cudaMemcpyAsync(gpu_kernel, kernel, KERNEL_SIZE * sizeof(float),
                  cudaMemcpyHostToDevice, 0);
  cudaMemcpyAsync(gpu_gx, gx, FILTER_SIZE * FILTER_SIZE * sizeof(float),
                  cudaMemcpyHostToDevice, 0);
  cudaMemcpyAsync(gpu_gy, gy, FILTER_SIZE * FILTER_SIZE * sizeof(float),
                  cudaMemcpyHostToDevice, 0);

  cudaStreamSynchronize(0);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  fprintf(stderr, " > CudaMem time: %f\n", elapsedTime / 1000.0f);

  // Step 1 - Grayscale
  cudaEventRecord(start, 0);

  gpu_grayscale_magic<<<(size + 511) / 512, 512>>>(
      input_image, gray_image, width, height, size, num_color);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  fprintf(stderr, " > Grayscale time: %f\n", elapsedTime / 1000.0f);

  // Step2 - Smooth
  cudaEventRecord(start, 0);
  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
  dim3 num_blocks(
      (width + BLOCK_SIZE - KERNEL_SIZE) / (BLOCK_SIZE - KERNEL_SIZE + 1),
      (height + BLOCK_SIZE - KERNEL_SIZE) / (BLOCK_SIZE - KERNEL_SIZE + 1));
  smooth_gaussian<<<num_blocks, block_size>>>(gray_image, smooth_image,
                                              gpu_kernel, width, height, size);
  // smooth_gaussian<<<
  //     ((width + KERNEL_SIZE - 1) * (height + KERNEL_SIZE - 1) + 1023) / 1024,
  //     blockNum>>>(gray_image, smooth_image, gpu_kernel, width, height, size);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  fprintf(stderr, " > Smooth time: %f\n", elapsedTime / 1000.0f);

  // // Step3 - Sobel
  cudaEventRecord(start, 0);

  float *gradient_matrix = gray_image;
  sobel<<<(size + 511) / 512, 512>>>(smooth_image, gradient_matrix,
                                     direction_matrix, gpu_gx, gpu_gy, width,
                                     height, size);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  fprintf(stderr, " > Sobel time: %f\n", elapsedTime / 1000.0f);

  // Step 4 - Gradient Magnitude Thresholding
  cudaEventRecord(start, 0);

  grad_mag_thresh<<<(size + 511) / 512, 512>>>(
      gradient_matrix, direction_matrix, width, height, size);
  float *thresh_img = direction_matrix;

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  fprintf(stderr, " > Gradient Magnitude Thresholding time: %f\n",
          elapsedTime / 1000.0f);

  // // Step 5 - Double Thresholding
  cudaEventRecord(start, 0);

  double_thresh(thresh_img);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  fprintf(stderr, " > Double Thresholding time: %f\n", elapsedTime / 1000.0f);

  // Step 6 - Hysteresis
  cudaEventRecord(start, 0);

  cast_int<<<(size + 511) / 512, 512>>>(
      thresh_img, input_image, width, height,
      size); // reuse input_image as output cache
  cudaDeviceSynchronize();
  cudaMemcpy(out_image, input_image, size * sizeof(uint8_t),
             cudaMemcpyDeviceToHost);
  hysteresis(out_image);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  fprintf(stderr, " > Hysteresis time: %f\n", elapsedTime / 1000.0f);

  return;
}

__global__ void gpu_grayscale_magic(uint8_t *image, float *out_image, int width,
                                    int height, int size, int num_color) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < size) {
    int r = idx / width;
    int c = idx % width;
    out_image[r * width + c] =
        0.299f * image[r * width * num_color + c * num_color] +
        0.587f * image[r * width * num_color + c * num_color + 1] +
        0.114f * image[r * width * num_color + c * num_color + 2];
  }
}

void generate_gaussian_kernel(float *kernel) {
  int half_size = KERNEL_SIZE / 2;
  float sum = 0.0f;

  // Compute the 1D Gaussian filter
  for (int i = -half_size; i <= half_size; ++i) {
    float value = exp(-(i * i) / (2.0f * SIGMA * SIGMA));
    kernel[i + half_size] = value;
    sum += value;
  }

  // Normalize the filter
  for (int i = 0; i < KERNEL_SIZE; ++i) {
    kernel[i] /= sum;
  }
}

__global__ void smooth_gaussian(float *input, float *output, float *kernel,
                                int width, int height, int size) {
  const int half_size = KERNEL_SIZE / 2;
  const int effective_block_size = BLOCK_SIZE - 2 * half_size;
  __shared__ float BLOCK[BLOCK_SIZE][BLOCK_SIZE];
  // Thread indices
  const int tr = threadIdx.y;
  const int tc = threadIdx.x;
  const int br = blockIdx.y;
  const int bc = blockIdx.x;

  // Global indices
  const int r = br * effective_block_size + tr;
  const int c = bc * effective_block_size + tc;

  BLOCK[tr][tc] = input[min(max(r - half_size, 0), height - 1) * width +
                        min(max(c - half_size, 0), width - 1)];
  __syncthreads();

  if (r < height && c < width) {
    if (tc < effective_block_size &&
        br * effective_block_size + tr < height + half_size &&
        bc * effective_block_size + tc < width) {
      float value = 0.0f;
      for (int kj = 0; kj < KERNEL_SIZE; ++kj) {
        value += BLOCK[tr][tc + kj] * kernel[kj];
      }

      BLOCK[tr][tc] = value;
    }
    __syncthreads();
    if (tr < effective_block_size && tc < effective_block_size &&
        br * effective_block_size + tr < height &&
        bc * effective_block_size + tc < width) {
      float value = 0.0f;

      for (int ki = 0; ki < KERNEL_SIZE; ++ki) {
        value += BLOCK[tr + ki][tc] * kernel[ki];
      }

      output[r * width + c] = value;
    }
  }
}

void generate_sobel_filter(float *filter, bool is_x_direction) {
  for (int i = -HALF_FILTER_SIZE; i <= HALF_FILTER_SIZE; ++i) {
    for (int j = -HALF_FILTER_SIZE; j <= HALF_FILTER_SIZE; ++j) {
      if (i != 0 || j != 0) {
        filter[(i + HALF_FILTER_SIZE) * FILTER_SIZE + j + HALF_FILTER_SIZE] =
            static_cast<float>(is_x_direction ? j : -i) / (i * i + j * j);
      } else {
        filter[(i + HALF_FILTER_SIZE) * FILTER_SIZE + j + HALF_FILTER_SIZE] = 0;
      }
    }
  }
}

__global__ void sobel(float *smooth_image, float *gradient_matrix,
                      float *direction_matrix, float *gpu_gx, float *gpu_gy,
                      int width, int height, int size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < size) {
    int i = idx / width;
    int j = idx % width;

    if (i < height && j < HALF_FILTER_SIZE) {
      gradient_matrix[i * width + j] = 0;
      gradient_matrix[i * width + width - 1 - j] = 0;
    }

    if (j < width && i < HALF_FILTER_SIZE) {
      gradient_matrix[i * width + j] = 0;
      gradient_matrix[(height - 1 - i) * width + j] = 0;
    }

    if (i >= HALF_FILTER_SIZE && i < height - HALF_FILTER_SIZE &&
        j >= HALF_FILTER_SIZE && j < width - HALF_FILTER_SIZE) {
      float x = 0, y = 0;
      // Convolve with the Sobel kernels
      for (int ki = -HALF_FILTER_SIZE; ki <= HALF_FILTER_SIZE; ++ki) {
        for (int kj = -HALF_FILTER_SIZE; kj <= HALF_FILTER_SIZE; ++kj) {
          int pixel_value = smooth_image[(i + ki) * width + j + kj];
          int kernel_idx =
              (ki + HALF_FILTER_SIZE) * FILTER_SIZE + (kj + HALF_FILTER_SIZE);

          x += pixel_value * gpu_gx[kernel_idx];
          y += pixel_value * gpu_gy[kernel_idx];
        }
      }

      // Compute gradient magnitude and direction
      gradient_matrix[i * width + j] = sqrt(y * y + x * x);
      direction_matrix[i * width + j] = atan(y / x);
    }
  }
}

__global__ void grad_mag_thresh(float *gradient_matrix, float *direction_matrix,
                                int width, int height, int size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < size) {
    int i = idx / width;
    int j = idx % width;

    if (i >= HALF_FILTER_SIZE && i < height - HALF_FILTER_SIZE &&
        j >= HALF_FILTER_SIZE && j < width - HALF_FILTER_SIZE) {
      float *dir = &direction_matrix[i * width + j];
      float mag = gradient_matrix[i * width + j];

      if (mag == 0) {
        *dir = 0;
      } else {
        float dir1 = 0;
        // this is maybe a few cycles slower than optimal
        // E <-> W
        if (-M_PI / 8 <= *dir && *dir < M_PI / 8) {
          for (int k = 1; k <= HALF_FILTER_SIZE; ++k) {
            dir1 = max(dir1, max(gradient_matrix[i * width + j + k],
                                 gradient_matrix[i * width + j - k]));
          }
        }
        // NE <-> SW
        else if (M_PI / 8 <= *dir && *dir < 3 * M_PI / 8) {
          for (int k = 1; k <= HALF_FILTER_SIZE; ++k) {
            dir1 = max(dir1, max(gradient_matrix[(i - k) * width + j + k],
                                 gradient_matrix[(i + k) * width + j - k]));
          }
        }
        // SE <-> NW
        else if (-3 * M_PI / 8 <= *dir && *dir < -M_PI / 8) {
          for (int k = 1; k <= HALF_FILTER_SIZE; ++k) {
            dir1 = max(dir1, max(gradient_matrix[(i + k) * width + j + k],
                                 gradient_matrix[(i - k) * width + j - k]));
          }
        }
        // N <-> S
        else if ((3 * M_PI / 8 <= *dir && *dir <= M_PI / 2) ||
                 (-M_PI / 2 <= *dir && *dir < -3 * M_PI / 8)) {
          for (int k = 1; k <= HALF_FILTER_SIZE; ++k) {
            dir1 = max(dir1, max(gradient_matrix[(i + k) * width + j],
                                 gradient_matrix[(i - k) * width + j]));
          }
        }

        if (max(mag, dir1) == mag) {
          *dir = mag;
        } else {
          *dir = 0;
        }
      }
    }
  }
}

// code borrow from https://cuvilib.com/Reduction.pdf
__global__ void max_kernel(const float *image, float *reduced_max, int size) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data into shared memory
  sdata[tid] = (idx < size) ? image[idx] : FLT_MIN;
  __syncthreads();

  // Perform reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] = max(sdata[tid], sdata[tid + stride]);
    }
    __syncthreads();
  }

  // Write the block's result to global memory
  if (tid == 0) {
    reduced_max[blockIdx.x] = sdata[0];
  }
}

__global__ void double_thresh_kernel(float *image, int width, int height,
                                     int size, float high_thresh,
                                     float low_thresh) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < size) {
    int i = idx / width;
    int j = idx % width;

    if (i >= 1 && i < height - 1 && j >= 1 && j <= width - 1) {
      float *cur = &image[i * width + j];
      if (*cur < low_thresh) {
        *cur = 0;
      } else if (*cur <= high_thresh) {
        *cur = WEAK;
      } else {
        *cur = STRONG;
      }
    }
  }
}

void double_thresh(float *image) {
  const int num_block = (size + 511) / 512;
  float *h_reduced_max = new float[num_block];

  float *reduced_max;
  cudaMalloc(&reduced_max, num_block * sizeof(float));
  max_kernel<<<num_block, 512, 512 * sizeof(float)>>>(image, reduced_max, size);
  cudaMemcpy(h_reduced_max, reduced_max, num_block * sizeof(float),
             cudaMemcpyDeviceToHost);

  float img_max = FLT_MIN;
  for (int i = 0; i < num_block; ++i) {
    img_max = std::max(img_max, h_reduced_max[i]);
  }

  delete[] h_reduced_max;
  float high_thresh = img_max * HIGH_THRESH_RATIO;
  float low_thresh = high_thresh * LOW_THRESH_RATIO;
  double_thresh_kernel<<<num_block, 512>>>(image, width, height, size,
                                           high_thresh, low_thresh);
}

__global__ void cast_int(float *input, uint8_t *output, int width, int height,
                         int size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < size) {
    output[idx] = static_cast<uint8_t>(input[idx]);
  }
}

void hysteresis(uint8_t *output) {
  for (int i = 1; i < height - 1; ++i) {
    for (int j = 1; j < width - 1; ++j) {

      if (output[index_nc1(i, j, width)] == WEAK) {
        for (int a = i - 1; a < i + 2; ++a) {
          for (int b = j - 1; b < j + 2; ++b) {
            if (output[index_nc1(a, b, width)] == STRONG) {
              output[index_nc1(i, j, width)] = STRONG;
              goto out;
            }
          }
        }
        output[index_nc1(i, j, width)] = 0;
      }

    out:;
    }
  }
}