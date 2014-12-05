#include "clhelp.h"
#include <opencv2/core/core.hpp>

using namespace cv;

class Batch {
  public:
    int batch_size;
    int rows;
    int cols;
    int depth;
    float* data;
    Batch(int b, int d, int r, int c): batch_size(b), rows(r), cols(c), depth(d) {
      data = new float[r * c * b * d]();
    };
};

class GpuBatch {
  public:
    cl_mem buf;
    int batch_size;
    int rows;
    int cols;
    int depth;

    GpuBatch(Batch b, cl_vars_t cv) {
      batch_size = b.batch_size;
      rows = b.rows;
      cols = b.cols;
      depth = b.depth;
      cl_int err = CL_SUCCESS;
      int size = batch_size * rows * cols * depth * sizeof(float);
      buf = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
                                      size, NULL, &err);
      CHK_ERR(err);
      err = clEnqueueWriteBuffer(cv.commands, buf, true, 0,
                                 size, b.data, 0, NULL, NULL);
      CHK_ERR(err);
      err = clFinish(cv.commands);
    };
};

class Weights {
  public:
    float* data;
    int rows;
    int cols;
    int r;
    int depth;
    int num_sets;
  Weights(int n_sets, int radius, int depth, float range): depth(depth) {
    num_sets = n_sets;
    rows = (radius * 2 + 1);
    cols = (radius * 2 + 1);
    data = new float[num_sets * depth * rows * cols];
    for (int n = 0; n < num_sets; n++) {
      for (int z = 0; z < depth; z++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < (cols); j++) {
                data[n * depth * rows * cols + z * rows * cols + i * cols + j] = (
                    (float) rand() / (float) RAND_MAX) * (range * 2 + 1) - range;
            }
        }
      }
    }
  }
};

class GpuWeights {
  public:
    cl_mem buf;
    int rows;
    int cols;
    int r;
    int depth;
    int num_sets;
  GpuWeights(Weights w, cl_vars_t cv) {
    cl_int err = CL_SUCCESS;
    buf = clCreateBuffer(cv.context, CL_MEM_READ_ONLY, w.rows * w.cols * w.num_sets *
        sizeof(float), NULL, &err);
    CHK_ERR(err);
    err = clEnqueueWriteBuffer(cv.commands, buf, true, 0, w.rows * w.cols * w.num_sets *
        sizeof(float), w.data, 0, NULL, NULL);
    CHK_ERR(err);
    depth = w.depth;
    rows = w.rows;
    cols = w.cols;
    r = w.r;
    num_sets = w.num_sets;
  }
};
