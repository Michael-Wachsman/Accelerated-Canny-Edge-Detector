#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <time.h>

#include "../common/solver.hpp"

#define BLOCK_SIZE 128

#define KERNEL_SIZE 7
#define SIGMA 1.4

#define FILTER_SIZE 5
#define HALF_FILTER_SIZE FILTER_SIZE / 2

#define NORMAL 1.0f / (2.0f * M_PI * SIGMA * SIGMA)
#define LOW_THRESH_RATIO .05
#define HIGH_THRESH_RATIO .2
#define WEAK 25
#define STRONG 255

int size, width, height;
float *gray_image;
float *smooth_image;
float *direction_matrix;

void free() {
  delete[] smooth_image;
  delete[] gray_image;
  delete[] direction_matrix;
}

void generate_gaussian_kernel(float *kernel);
void apply_gaussian_filter(float *input, float *output, float *kernel);
void smooth_gaussian(float *input, float *output);

void generate_sobel_filter(float *filter, bool is_x_direction);
void apply_sobel_filter(float *smooth_image, float *gradient_matrix,
                        float *direction_matrix, float *gx, float *gy);
void sobel(float *smooth_image, float *gradient_matrix,
           float *direction_matrix);
float *grad_mag_thresh(float *gradient_matrix, float *direction_matrix);
void double_thresh(float *image);
void hysteresis(float *input, uint8_t *output);

void findEdges(uint8_t *image, uint8_t *out_image, int w, int h,
               int num_color) {
  width = w;
  height = h;
  size = width * height;

  // Step 1 - Grayscale
  clock_t gray_start = clock();
  gray_image = new float[size];
  grayscale_magic(image, gray_image, width, height, num_color);
  clock_t gray_end = clock();
  fprintf(stderr, " > Grayscale time: %f\n",
          (double)(gray_end - gray_start) / CLOCKS_PER_SEC);

  // Step2 - Smooth
  clock_t smooth_start = clock();
  smooth_image = new float[size];
  smooth_gaussian(gray_image, smooth_image);
  clock_t smooth_end = clock();
  fprintf(stderr, " > Smooth time: %f\n",
          (double)(smooth_end - smooth_start) / CLOCKS_PER_SEC);

  // Step3 - Sobel
  clock_t sobel_start = clock();
  float *gradient_matrix = gray_image;
  direction_matrix = new float[size];
  sobel(smooth_image, gradient_matrix, direction_matrix);
  clock_t sobel_end = clock();
  fprintf(stderr, " > Sobel time: %f\n",
          (double)(sobel_end - sobel_start) / CLOCKS_PER_SEC);

  // Step 4 - Gradient Magnitude Thresholding
  clock_t gmt_start = clock();
  float *thresh_img = grad_mag_thresh(gradient_matrix, direction_matrix);
  clock_t gmt_end = clock();
  fprintf(stderr, " > Gradient Magnitude Thresholding time: %f\n",
          (double)(gmt_end - gmt_start) / CLOCKS_PER_SEC);

  // Step 5 - Double Thresholding
  clock_t dt_start = clock();
  double_thresh(thresh_img);
  clock_t dt_end = clock();
  fprintf(stderr, " > Double Thresholding time: %f\n",
          (double)(dt_end - dt_start) / CLOCKS_PER_SEC);

  // Step 6 - Hysteresis
  clock_t hys_start = clock();
  hysteresis(thresh_img, out_image);
  clock_t hys_end = clock();
  fprintf(stderr, " > Hysteresis time: %f\n",
          (double)(hys_end - hys_start) / CLOCKS_PER_SEC);

  return;
}

void smooth_gaussian(float *input, float *output) {
  float kernel[KERNEL_SIZE * KERNEL_SIZE];

  generate_gaussian_kernel(kernel);
  apply_gaussian_filter(input, output, kernel);
}

void generate_gaussian_kernel(float *kernel) {
  int half_size = KERNEL_SIZE / 2;
  float sum = 0.0f;

  for (int i = -half_size; i <= half_size; ++i) {
    for (int j = -half_size; j <= half_size; ++j) {
      float value = NORMAL * exp(-(i * i + j * j) / (2.0f * SIGMA * SIGMA));
      kernel[(i + half_size) * KERNEL_SIZE + (j + half_size)] = value;
      sum += value;
    }
  }

  // Normalize the kernel
  for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; ++i) {
    kernel[i] /= sum;
  }
}

void apply_gaussian_filter(float *input, float *output, float *kernel) {
  int half_size = KERNEL_SIZE / 2;
  const int effective_block_size = BLOCK_SIZE - 2 * half_size;
  float BLOCK[BLOCK_SIZE * BLOCK_SIZE]
      __attribute__((aligned(64)));

  for (int bi = 0; bi < height; bi += effective_block_size) {
    for (int bj = 0; bj < width; bj += effective_block_size) {

      for (int i = -half_size; i < effective_block_size + half_size; ++i) {
        for (int j = -half_size; j < effective_block_size + half_size; ++j) {
          int ni = std::clamp(bi + i, 0, height - 1);
          int nj = std::clamp(bj + j, 0, width - 1);
          BLOCK[(i + half_size) * BLOCK_SIZE + (j + half_size)] =
              input[ni * width + nj];
        }
      }

      for (int i = 0; i < effective_block_size && bi + i < height; ++i) {
        for (int j = 0; j < effective_block_size && bj + j < width; ++j) {
          float value = 0.0f;

          for (int ki = 0; ki < KERNEL_SIZE; ++ki) {
            for (int kj = 0; kj < KERNEL_SIZE; ++kj) {
              value += BLOCK[(i + ki) * BLOCK_SIZE + (j + kj)] *
                       kernel[ki * KERNEL_SIZE + kj];
            }
          }

          output[(bi + i) * width + (bj + j)] = value;
        }
      }
    }
  }
}

void sobel(float *smooth_image, float *gradient_matrix,
           float *direction_matrix) {
  float gx[FILTER_SIZE * FILTER_SIZE];
  float gy[FILTER_SIZE * FILTER_SIZE];
  generate_sobel_filter(gx, true);
  generate_sobel_filter(gy, false);
  apply_sobel_filter(smooth_image, gradient_matrix, direction_matrix, gx, gy);
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

void apply_sobel_filter(float *smooth_image, float *gradient_matrix,
                        float *direction_matrix, float *gx, float *gy) {
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < HALF_FILTER_SIZE; ++j) {
      gradient_matrix[index_nc1(i, j, width)] = 0;
      gradient_matrix[index_nc1(i, width - 1 - j, width)] = 0;
    }
  }

  for (int j = 0; j < width; ++j) {
    for (int i = 0; i < HALF_FILTER_SIZE; ++i) {
      gradient_matrix[index_nc1(i, j, width)] = 0;
      gradient_matrix[index_nc1(height - 1 - i, j, width)] = 0;
    }
  }

  float x, y;

  // const int effective_block_size = BLOCK_SIZE - 2 * HALF_FILTER_SIZE;
  // static float BLOCK[BLOCK_SIZE * BLOCK_SIZE]
  //     __attribute__((aligned(BLOCK_SIZE)));

  // for (int bi = HALF_FILTER_SIZE; bi < height - HALF_FILTER_SIZE;
  //      bi += effective_block_size) {
  //   for (int bj = HALF_FILTER_SIZE; bj < width - HALF_FILTER_SIZE;
  //        bj += effective_block_size) {

  //     for (int i = 0; i < BLOCK_SIZE; ++i) {
  //       std::memcpy(BLOCK + i * BLOCK_SIZE,
  //                   smooth_image + (bi + i - HALF_FILTER_SIZE) * width + bj -
  //                       HALF_FILTER_SIZE,
  //                   BLOCK_SIZE * sizeof(float));
  //     }

  //     const int max_i = std::min(effective_block_size, height - HALF_FILTER_SIZE - bi);
  //     const int max_j = std::min(effective_block_size, width - HALF_FILTER_SIZE - bj);
  //     for (int i = 0; i < max_i; ++i) {
  //       for (int j = 0; j < max_j; ++j) {
  //         x = 0;
  //         y = 0;

  //         for (int ki = 0; ki < FILTER_SIZE; ++ki) {
  //           for (int kj = 0; kj < FILTER_SIZE; ++kj) {
  //             int pixel_value = BLOCK[(i + ki) * BLOCK_SIZE + (j + kj)];
  //             int kernel_idx = ki * FILTER_SIZE + kj;

  //             x += pixel_value * gx[kernel_idx];
  //             y += pixel_value * gy[kernel_idx];
  //           }
  //         }

  //         int idx = (bi + i) * width + (bj + j);
  //         gradient_matrix[idx] = std::sqrt(x * x + y * y);
  //         direction_matrix[idx] = std::atan(y / x);
  //       }
  //     }
  //   }
  // }

  for (int i = HALF_FILTER_SIZE; i < height - HALF_FILTER_SIZE; ++i) {
    for (int j = HALF_FILTER_SIZE; j < width - HALF_FILTER_SIZE; ++j) {
      x = 0;
      y = 0;

      for (int ki = -HALF_FILTER_SIZE; ki <= HALF_FILTER_SIZE; ++ki) {
        for (int kj = -HALF_FILTER_SIZE; kj <= HALF_FILTER_SIZE; ++kj) {
          int pixel_value = smooth_image[index_nc1(i + ki, j + kj, width)];
          int kernel_idx =
              (ki + HALF_FILTER_SIZE) * FILTER_SIZE + (kj +
              HALF_FILTER_SIZE);

          x += pixel_value * gx[kernel_idx];
          y += pixel_value * gy[kernel_idx];
        }
      }

      gradient_matrix[index_nc1(i, j, width)] = std::sqrt(y * y + x * x);
      direction_matrix[index_nc1(i, j, width)] = std::atan(y / x);
    }
  }
}

float *grad_mag_thresh(float *gradient_matrix, float *direction_matrix) {
  for (int i = HALF_FILTER_SIZE; i < height - HALF_FILTER_SIZE; i++) {
    for (int j = HALF_FILTER_SIZE; j < width - HALF_FILTER_SIZE; j++) {
      float *dir = &direction_matrix[index_nc1(i, j, width)];
      float mag = gradient_matrix[index_nc1(i, j, width)];

      if (mag == 0) {
        *dir = 0;
      } else {
        float dir1 = 0;
        // this is maybe a few cycles slower than optimal
        // E <-> W
        if (-M_PI / 8 <= *dir && *dir < M_PI / 8) {
          for (int k = 1; k <= HALF_FILTER_SIZE; ++k) {
            dir1 = std::max(
                dir1, std::max(gradient_matrix[index_nc1(i, j + k, width)],
                               gradient_matrix[index_nc1(i, j - k, width)]));
          }
        }
        // NE <-> SW
        else if (M_PI / 8 <= *dir && *dir < 3 * M_PI / 8) {
          for (int k = 1; k <= HALF_FILTER_SIZE; ++k) {
            dir1 = std::max(
                dir1,
                std::max(gradient_matrix[index_nc1(i - k, j + k, width)],
                         gradient_matrix[index_nc1(i + k, j - k, width)]));
          }
        }
        // SE <-> NW
        else if (-3 * M_PI / 8 <= *dir && *dir < -M_PI / 8) {
          for (int k = 1; k <= HALF_FILTER_SIZE; ++k) {
            dir1 = std::max(
                dir1,
                std::max(gradient_matrix[index_nc1(i + k, j + k, width)],
                         gradient_matrix[index_nc1(i - k, j - k, width)]));
          }
        }
        // N <-> S
        else if ((3 * M_PI / 8 <= *dir && *dir <= M_PI / 2) ||
                 (-M_PI / 2 <= *dir && *dir < -3 * M_PI / 8)) {
          for (int k = 1; k <= HALF_FILTER_SIZE; ++k) {
            dir1 = std::max(
                dir1, std::max(gradient_matrix[index_nc1(i + k, j, width)],
                               gradient_matrix[index_nc1(i - k, j, width)]));
          }
        }

        if (std::max(mag, dir1) == mag) {
          *dir = mag;
        } else {
          *dir = 0;
        }
      }
    }
  }

  return direction_matrix;
}

void double_thresh(float *image) {
  float img_max = image[0];
  for (int i = 1; i < size; ++i) {
    img_max = std::max(img_max, image[i]);
  }
  float high_thresh = img_max * HIGH_THRESH_RATIO;
  float low_thresh = high_thresh * LOW_THRESH_RATIO;

  for (int i = 1; i < height - 1; ++i) {
    for (int j = 1; j < width - 1; ++j) {
      float *cur = &image[index_nc1(i, j, width)];
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

void hysteresis(float *input, uint8_t *output) {
  for (int i = 0; i < size; ++i) {
    output[i] = static_cast<int>(input[i]);
  }

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