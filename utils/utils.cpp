#include "utils.h"

void grayscale_avg(uint8_t * image, uint8_t * out_image, int width, int height, int num_color) {
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            out_image[index(r, c, 0, width, 1)] = std::min(image[index(r, c, 0, width, NUM_COLORS)], std::min(image[index(r, c, 1, width, NUM_COLORS)], image[index(r, c, 2, width, NUM_COLORS)]));
            // out_image[index(r, c, 0, width, 1)] = std::max(image[index(r, c, 0, width, NUM_COLORS)], std::max(image[index(r, c, 1, width, NUM_COLORS)], image[index(r, c, 2, width, NUM_COLORS)]));
            // out_image[index(r, c, 0, width, 1)] = (image[index(r, c, 0, width, NUM_COLORS)] + image[index(r, c, 1, width, NUM_COLORS)] + image[index(r, c, 2, width, NUM_COLORS)]) / 3;
        }
    }
}

void grayscale_magic(uint8_t * image, float * out_image, int width, int height, int num_color) {
    #ifdef PARALLEL
    #pragma omp parallel for 
    #endif
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            out_image[index_nc1(r, c, width)] = 0.299f * image[index(r, c, 0, width, num_color)] + 
                               0.587f * image[index(r, c, 1, width, num_color)] + 
                               0.114f * image[index(r, c, 2, width, num_color)];
        }
    }
}

void grayscale_red(uint8_t * image, uint8_t * out_image, int width, int height, int num_color) {
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            out_image[index(r, c, 0, width, 1)] = image[index(r, c, 0, width, NUM_COLORS)];
        }
    }
}

// made by chat and verified by me lol
//only gets the cardinal directions and the diagonals THAT MATTER
float superSimpleFastAtan2(float x, float y){
    float abs_x = std::fabs(x);
    float abs_y = std::fabs(y);

    
    if (y >= 0) { // Top half-plane
        if (x >= 0) { // Quadrant I
            if (abs_x > 2 * abs_y) {
                return 0.0; // Cone centered on 0° (0 radians)
            } else if (abs_y > 2 * abs_x) {
                return M_PI / 2 - .0001; // Cone centered on 90° (pi/2 radians)
            } else {
                return M_PI / 4; // Cone centered on 45° (pi/4 radians)
            }
        } else { // Quadrant II
            if (abs_y > 2 * abs_x) {
                return M_PI / 2  - .0001; // Cone centered on 90° (pi/2 radians)
            } else if (abs_x > 2 * abs_y) {
                return 0; // Cone centered on 180° (pi radians)
            } else {
                return -M_PI / 4; // Cone centered on 135° (3pi/4 radians)
            }
        }
    } else { // Bottom half-plane
        if (x <= 0) { // Quadrant III
            if (abs_x > 2 * abs_y) {
                return 0; // Cone centered on 180° (pi radians)
            } else if (abs_y > 2 * abs_x) {
                return -M_PI / 2  + .0001; // Cone centered on 270° (-pi/2 radians)
            } else {
                return  M_PI / 4; // Cone centered on 225° (-3pi/4 radians)
            }
        } else { // Quadrant IV
            if (abs_y > 2 * abs_x) {
                return -M_PI / 2  + .0001; // Cone centered on 270° (-pi/2 radians)
            } else if (abs_x > 2 * abs_y) {
                return 0.0; // Cone centered on 0° (0 radians)
            } else {
                return -M_PI / 4; // Cone centered on 315° (-pi/4 radians)
            }
        }
    }

    return 0;
}
