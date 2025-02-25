#include <cmath>
#include <cstdio>
#include <time.h>

#include "../common/solver.hpp"

#define EDGE_THRESHOLD 75

uint8_t *gray_image;

void free() { delete[] gray_image; }

void sobel(uint8_t *input, uint8_t *output, int nx, int ny);

void findEdges(uint8_t *image, uint8_t *out_image, int width, int height,
               int num_color) {
  // Step 1 - Grayscale
  clock_t gray_start = clock();
  uint8_t *gray_image = new uint8_t[width * height];
  grayscale_avg(image, gray_image, width, height, num_color);
  clock_t gray_end = clock();
  fprintf(stderr, " > Grayscale time: %f\n",
          (double)(gray_end - gray_start) / CLOCKS_PER_SEC);

  // Step 2 - Sobel
  clock_t sobel_start = clock();
  sobel(gray_image, out_image, width, height);
  clock_t sobel_end = clock();
  fprintf(stderr, " > Sobel time: %f\n",
          (double)(sobel_end - sobel_start) / CLOCKS_PER_SEC);

  return;
}

void sobel(uint8_t *pixels, uint8_t *output, int nx, int ny) {
  int nc = 1;
  // Sobel Horizontal Mask
  static int GX00, GX01, GX02, GX10, GX11, GX12, GX20, GX21, GX22, GY00, GY01,
      GY02, GY10, GY11, GY12, GY20, GY21, GY22;
  // Two arrays to store values for parallelization purposes
  int **TMPX = new int *[ny];
  int **TMPY = new int *[ny];

  for (int i = 0; i < ny; i++) {
    TMPY[i] = new int[nx];
    TMPX[i] = new int[nx];
  }

  // Sobel Horizontal Mask
  GX00 = 1;
  GX01 = 0;
  GX02 = -1;
  GX10 = 2;
  GX11 = 0;
  GX12 = -2;
  GX20 = 1;
  GX21 = 0;
  GX22 = -1;

  // Sobel Vertical Mask
  GY00 = 1;
  GY01 = 2;
  GY02 = 1;
  GY10 = 0;
  GY11 = 0;
  GY12 = 0;
  GY20 = -1;
  GY21 = -2;
  GY22 = -1;

  int MAG;
  {
    for (int i = 0; i < ny; i++) {
      for (int j = 0; j < nx; j++) {
        TMPY[i][j] = 0;
        TMPX[i][j] = 0;
      }
    }

    for (int i = 0; i < ny; i++) {
      for (int j = 0; j < nx; j++) {
        // setting the pixels around the border to 0, because the Sobel kernel
        // cannot be allied to them
        if ((i == 0) || (i == ny - 1) || (j == 0) || (j == nx - 1)) {
          TMPX[i][j] = 0;
          TMPY[i][j] = 0;
        } else {
          TMPY[i][j] += pixels[index(i - 1, j - 1, 0, nx, nc)] * GY00 +
                        pixels[index(i, j - 1, 0, nx, nc)] * GY10 +
                        pixels[index(i + 1, j - 1, 0, nx, nc)] * GY20 +
                        pixels[index(i - 1, j, 0, nx, nc)] * GY01 +
                        pixels[index(i, j, 0, nx, nc)] * GY11 +
                        pixels[index(i + 1, j, 0, nx, nc)] * GY21 +
                        pixels[index(i - 1, j, 0, nx, nc)] * GY02 +
                        pixels[index(i, j, 0, nx, nc)] * GY12 +
                        pixels[index(i + 1, j, 0, nx, nc)] * GY22;

          TMPX[i][j] += pixels[index(i - 1, j - 1, 0, nx, nc)] * GX00 +
                        pixels[index(i, j - 1, 0, nx, nc)] * GX10 +
                        pixels[index(i + 1, j - 1, 0, nx, nc)] * GX20 +
                        pixels[index(i - 1, j, 0, nx, nc)] * GX01 +
                        pixels[index(i, j, 0, nx, nc)] * GX11 +
                        pixels[index(i + 1, j, 0, nx, nc)] * GX21 +
                        pixels[index(i - 1, j, 0, nx, nc)] * GX02 +
                        pixels[index(i, j, 0, nx, nc)] * GX12 +
                        pixels[index(i + 1, j, 0, nx, nc)] * GX22;
        }
      }
    }

    for (int i = 0; i < ny; i++) {
      for (int j = 0; j < nx; j++) {
        // Gradient magnitude
        MAG = sqrt(TMPX[i][j] * TMPX[i][j] + TMPY[i][j] * TMPY[i][j]);

        // Apply threshold to gradient
        if (MAG > EDGE_THRESHOLD)
          MAG = 255;
        else
          MAG = 0;

        // setting the new pixel value
        output[index(i, j, 0, nx, 1)] = MAG;
      }
    }
  }

  for (int i = 0; i < ny; ++i)
    delete[] TMPY[i];
  for (int i = 0; i < ny; ++i)
    delete[] TMPX[i];
  delete[] TMPY;
  delete[] TMPX;
}