#include "clhelp.h"
#include <opencv2/core/core.hpp>

using namespace cv;

class GpuMat {
  public:
    cl_mem buf;
    int rows;
    int cols;
    GpuMat(Mat input, cl_vars_t cv);
    GpuMat(cl_mem input, int r, int c): buf(input), rows(r), cols(c) {};
};

GpuMat::GpuMat(Mat input, cl_vars_t cv) {

    cl_int err = CL_SUCCESS;
    int size = input.total() * input.elemSize();
    cl_mem ocl_buf = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
                                    size, NULL, &err);
    CHK_ERR(err);
    err = clEnqueueWriteBuffer(cv.commands, ocl_buf, true, 0,
                               size, (float*)input.data, 0,
                               NULL, NULL);
    CHK_ERR(err);
    rows = input.rows;
    cols = input.cols;
    buf = ocl_buf;
}

class Batch {
  public:
    int batch_size;
    int rows;
    int cols;
    float* data;
    Batch(int b, int r, int c): batch_size(b), rows(r), cols(c) {
      data = new float[r * c * b]();
    };
};

class GpuBatch {
  public:
    cl_mem buf;
    int batch_size;
    int rows;
    int cols;

    GpuBatch(Batch b, cl_vars_t cv) {
      batch_size = b.batch_size;
      rows = b.rows;
      cols = b.cols;
      cl_int err = CL_SUCCESS;
      int size = batch_size * rows * cols * sizeof(float);
      cl_mem ocl_buf = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
                                      size, NULL, &err);
      CHK_ERR(err);
      err = clEnqueueWriteBuffer(cv.commands, ocl_buf, true, 0,
                                 size, (float*)b.data, 0, NULL, NULL);
      CHK_ERR(err);
      buf = ocl_buf;
    };
};
