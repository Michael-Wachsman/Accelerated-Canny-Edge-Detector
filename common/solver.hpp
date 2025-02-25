#ifndef SOLVER_H
#define SOLVER_H
#include <cstdint>

#include "../utils/utils.h"

void free();
void findEdges(uint8_t *image, uint8_t *out_image, int width, int height,
               int num_color);

#endif