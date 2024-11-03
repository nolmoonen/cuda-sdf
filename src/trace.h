#pragma once

#include "camera.h"

typedef unsigned int uint;
typedef unsigned char uchar;

/// Number of iterations of the Menger sponge.
#define ITER_COUNT 20

/// Number of ray bounces.
#define MIN_BOUNCE_COUNT 0
#define MAX_BOUNCE_COUNT 80

/// Whether to use a naive implementation.
//#define NAIVE

void generate(
        uint size_x, uint size_y, uint sample_count, const char *filename);
