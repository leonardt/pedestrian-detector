#ifndef __CONVOLUTION_H
#define __CONVOLUTION_H

#include <iostream>
#include <assert.h>
#include <limits>
#include <vector>
#include <math.h>

#include "opencv2/video/tracking.hpp"

void convolve(float* input, float* output, float* weights, int n, int r);

#endif