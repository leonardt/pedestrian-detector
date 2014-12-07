#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cassert>
#include <cmath>

// #include "clhelp.h"
#include "types.cpp"
//#include "convolution.h"
#include <iostream>


#include <opencv2/core/core.hpp>

using namespace cv;

void ocl_conv(GpuBatch input, GpuBatch output, vector<GpuWeights> weights_sets, 
              int r, cl_vars_t cv, cl_kernel conv, cl_uint device) {
  cl_int err = CL_SUCCESS;
  for (size_t i = 0; i < weights_sets.size(); i++) {
    size_t global_work_size[3] = {static_cast<size_t>(input.rows), 
                                  static_cast<size_t>(input.cols), 
                                  static_cast<size_t>(input.batch_size)};
    size_t local_work_size[3] = {static_cast<size_t>(input.rows), 
                                 static_cast<size_t>(input.cols), 
                                 1};

    /* Set kernel arguments */

    err = clSetKernelArg(conv, 0, sizeof(cl_mem), &output.buf);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 1, sizeof(cl_mem), &input.buf);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 2, sizeof(cl_mem), &weights_sets[i].buf);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 3, sizeof(int), &input.rows);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 4, sizeof(int), &input.cols);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 5, sizeof(int), &input.depth);
    CHK_ERR(err);

    int size = weights_sets.size();
    err = clSetKernelArg(conv, 6, sizeof(int), &size);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 7, sizeof(int), &i);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 8, sizeof(float) * input.rows * input.cols, NULL);
    CHK_ERR(err);

    err = clEnqueueNDRangeKernel(cv.commands[device],
            conv,
            3, //2 work dimensions
            NULL,
            global_work_size,
            local_work_size,
            0,
            NULL,
            NULL
    );
    CHK_ERR(err);
  }
}
