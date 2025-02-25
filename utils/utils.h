#ifndef UTILS_H
#define UTILS_H

#include <cstdint>
#include <algorithm>
#include <cmath>

#define NUM_COLORS 3

inline int index(int r, int c, int color_idx, int width, int num_color) { return r * width * num_color + c * num_color + color_idx; }
inline int index_nc1(int r, int c, int width) {return r * width + c; }

void grayscale_avg(uint8_t * image, uint8_t * out_image, int width, int height, int num_color);

void grayscale_magic(uint8_t * image, float * out_image, int width, int height, int num_color);

void grayscale_red(uint8_t * image, uint8_t * out_image, int width, int height, int num_color);

float superSimpleFastAtan2(float x, float y);

#endif