#ifndef TYPES_H
#define TYPES_H
#include "clhelp.h"

using namespace cv;

class Batch {
  public:
    int batch_size;
    int rows;
    int cols;
    int depth;
    float* data;
    Batch(int b, int d, int r, int c);
};

class GpuBatch {
  public:
    cl_mem buf;
    int batch_size;
    int rows;
    int cols;
    int depth;

    GpuBatch(Batch b, cl_vars_t cv, cl_uint device);
};

class Weights {
  public:
    float* data;
    int rows;
    int cols;
    int depth;
  Weights(int radius, int depth, float range);
};

class GpuWeights {
  public:
    cl_mem buf;
    int rows;
    int cols;
    int depth;
  GpuWeights(Weights w, cl_vars_t cv, cl_uint device);
};
#endif
