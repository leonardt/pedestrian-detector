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

    GpuBatch(Batch b, cl_vars_t cv, cl_uint device) {
      batch_size = b.batch_size;
      rows = b.rows;
      cols = b.cols;
      depth = b.depth;
      cl_int err = CL_SUCCESS;
      int size = batch_size * rows * cols * depth * sizeof(float);
      buf = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
                                      size, NULL, &err);
      CHK_ERR(err);
      err = clEnqueueWriteBuffer(cv.commands[device], buf, true, 0,
                                 size, b.data, 0, NULL, NULL);
      CHK_ERR(err);
      err = clFinish(cv.commands[device]);
    };
};

class Weights {
  public:
    float* data;
    int rows;
    int cols;
    int depth;
  Weights(int radius, int depth, float range): depth(depth) {
    rows = (radius * 2 + 1);
    cols = (radius * 2 + 1);
    data = new float[depth * rows * cols]();
    if (range > 0) {
      for (int z = 0; z < depth; z++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[z * rows * cols + i * cols + j] = (
                    (float) rand() / (float) RAND_MAX) * ( (1 / range) * 2 + 1) - (1 / range);
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
    int depth;
  GpuWeights(Weights w, cl_vars_t cv, cl_uint device) {
    cl_int err = CL_SUCCESS;
    int size = w.depth * w.rows * w.cols * sizeof(float);
    buf = clCreateBuffer(cv.context, CL_MEM_READ_ONLY, size, NULL, &err);
    CHK_ERR(err);
    err = clEnqueueWriteBuffer(cv.commands[device], buf, true, 0, size, w.data, 0, NULL, NULL);
    CHK_ERR(err);
    depth = w.depth;
    rows = w.rows;
    cols = w.cols;
  }
};
