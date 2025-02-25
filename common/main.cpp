#include <cstdint>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <time.h>

#ifdef CUDA
#include <cuda_runtime.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "../utils/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../utils/stb_image_write.h"

#include "../utils/utils.h"
#include "solver.hpp"

int main(int argc, char **argv) {
  char input_file[256] = "images/input.jpg", output_file[256],
       output_txt_file[256];
  bool output = false;
  bool output_txt = false;

  int cur_arg = 1;
  int num_args = argc - 1;

  while (num_args > 0) {
    if (num_args == 1) {
      fprintf(stderr, "Missing argument value for %s\n", argv[cur_arg]);
      return 1;
    }

    if (strcmp(argv[cur_arg], "--output") == 0) {
      strcpy(output_file, argv[cur_arg + 1]);
      output = true;
    } else if (strcmp(argv[cur_arg], "--input") == 0) {
      strcpy(input_file, argv[cur_arg + 1]);
    } else if (strcmp(argv[cur_arg], "--output_txt") == 0) {
      strcpy(output_txt_file, argv[cur_arg + 1]);
      output_txt = true;
    } else {
      fprintf(stderr, "Unknown argument: %s\n", argv[cur_arg]);
      return 1;
    }

    cur_arg += 2;
    num_args -= 2;
  }

  // Read Image
  int width, height, num_color;
  uint8_t *image =
      stbi_load(input_file, &width, &height, &num_color, NUM_COLORS);
  fprintf(stderr,
          "Read image %s. Width: %dpx. Height: %dpx. Actual NumColor: %d. Read "
          "NumColor: %d.\n",
          input_file, width, height, num_color, NUM_COLORS);

#ifdef DETAIL_TIME
  // Edge Detection
  fprintf(stderr, "Edge detection Start:\n");
#endif

#ifdef CUDA
  cudaEvent_t s;
  cudaEventCreate(&s);
#endif

#ifdef PARALLEL
  double edge_start = omp_get_wtime();
#else
  clock_t edge_start = clock();
#endif

  uint8_t *edge_image = new uint8_t[width * height];
  findEdges(image, edge_image, width, height, NUM_COLORS);

#ifdef CUDA
  cudaEventSynchronize(s);
#endif

#ifdef PARALLEL
  double edge_end = omp_get_wtime();
  fprintf(stderr, "Edge detection time: %f\n", (edge_end - edge_start));
#else
  clock_t edge_end = clock();
  fprintf(stderr, "Edge detection time: %f\n",
          (double)(edge_end - edge_start) / CLOCKS_PER_SEC);
#endif

  // Write Output
  if (output) {
    stbi_write_jpg(output_file, width, height, 1, edge_image,
                   100); // 100 for best image quality
    fprintf(stderr, "Write image %s.\n", output_file);
  }

  if (output_txt) {
    int size = width * height;
    std::ofstream file(output_txt_file);
    for (int i = 0; i < size; ++i) {
      file << static_cast<int>(edge_image[i]);
      if (i < size - 1) {
        file << " ";
      }
    }
    file.close();
    fprintf(stderr, "Write txt %s.\n", output_txt_file);
  }

  // Free Memory
  clock_t free_start = clock();
  stbi_image_free(image);
  delete[] edge_image;
  free();
  clock_t free_end = clock();

  fprintf(stderr, "Free memory time: %f\n",
          (double)(free_end - free_start) / CLOCKS_PER_SEC);

#ifdef PARALLEL
  fprintf(stderr, "%f\n",
          (edge_end - edge_start) +
              (double)(free_end - free_start) / CLOCKS_PER_SEC);
#else
  fprintf(stderr, "%f\n",
          (double)(edge_end - edge_start + free_end - free_start) /
              CLOCKS_PER_SEC);
#endif

  return 0;
}
