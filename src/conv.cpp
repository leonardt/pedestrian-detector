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
              vector<float> bias, int r, cl_vars_t cv, cl_kernel conv, cl_uint device) {
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

    err = clSetKernelArg(conv, 3, sizeof(float), &bias[i]);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 4, sizeof(int), &input.rows);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 5, sizeof(int), &input.cols);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 6, sizeof(int), &input.depth);
    CHK_ERR(err);

    int size = weights_sets.size();
    err = clSetKernelArg(conv, 7, sizeof(int), &size);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 8, sizeof(int), &i);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 9, sizeof(float) * input.rows * input.cols, NULL);
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

void ocl_hidden_layer(GpuBatch input, GpuWeights weights, GpuWeights bias, 
                      GpuBatch output, GpuBatch v, int last_layer, cl_vars_t cv,
                      cl_kernel kern, cl_uint device) {
  cl_int err = CL_SUCCESS;
  size_t global_work_size[2] = {static_cast<size_t>(input.batch_size), 4};
  size_t local_work_size[2] = {128, 4};

  /* Set kernel arguments */

  err = clSetKernelArg(kern, 0, sizeof(cl_mem), &output.buf);
  CHK_ERR(err);

  err = clSetKernelArg(kern, 1, sizeof(cl_mem), &v.buf);
  CHK_ERR(err);

  err = clSetKernelArg(kern, 2, sizeof(cl_mem), &input.buf);
  CHK_ERR(err);

  err = clSetKernelArg(kern, 3, sizeof(cl_mem), &weights.buf);
  CHK_ERR(err);
  
  err = clSetKernelArg(kern, 4, sizeof(cl_mem), &bias.buf);
  CHK_ERR(err);
  
  err = clSetKernelArg(kern, 5, sizeof(int), &last_layer);
  CHK_ERR(err);
  
  int size = input.rows * input.cols * input.depth;

  err = clSetKernelArg(kern, 6, sizeof(int), &size);
  CHK_ERR(err);

  err = clSetKernelArg(kern, 7, 4 * sizeof(float), NULL);
  CHK_ERR(err);

  err = clEnqueueNDRangeKernel(cv.commands[device],
          kern,
          2,
          NULL,
          global_work_size,
          local_work_size,
          0,
          NULL,
          NULL
  );
  CHK_ERR(err);
}

void ocl_softmax(GpuBatch input, cl_vars_t cv, cl_kernel kern, cl_uint device) {
  cl_int err = CL_SUCCESS;
  size_t global_work_size[2] = {static_cast<size_t>(input.batch_size)};
  size_t local_work_size[2] = {512};

  /* Set kernel arguments */

  err = clSetKernelArg(kern, 0, sizeof(cl_mem), &input.buf);
  CHK_ERR(err);

  err = clSetKernelArg(kern, 1, sizeof(float) * 512, NULL);
  CHK_ERR(err);

  err = clEnqueueNDRangeKernel(cv.commands[device],
          kern,
          1,
          NULL,
          global_work_size,
          local_work_size,
          0,
          NULL,
          NULL
  );
  CHK_ERR(err);
}

void ocl_tanh(GpuBatch input, GpuBatch v, cl_vars_t cv, cl_kernel kern, cl_uint device) {
  cl_int err = CL_SUCCESS;
  size_t global_work_size[2] = {static_cast<size_t>(input.batch_size * input.rows * input.cols * input.depth)};
  size_t local_work_size[2] = {512};

  /* Set kernel arguments */

  err = clSetKernelArg(kern, 0, sizeof(cl_mem), &input.buf);
  CHK_ERR(err);

  err = clSetKernelArg(kern, 1, sizeof(cl_mem), &v.buf);
  CHK_ERR(err);

  err = clEnqueueNDRangeKernel(cv.commands[device],
          kern,
          1,
          NULL,
          global_work_size,
          local_work_size,
          0,
          NULL,
          NULL
  );
  CHK_ERR(err);
}

void ocl_ol_back(GpuBatch delta, GpuBatch actual, GpuBatch prev_out, GpuBatch prev_v, cl_kernel kern, cl_vars_t cv) {
  cl_int err = CL_SUCCESS;
  size_t global_work_size[2] = {static_cast<size_t>(delta.batch_size * delta.rows * delta.cols * delta.depth)};
  size_t local_work_size[2] = {512};

  /* Set kernel arguments */

  err = clSetKernelArg(kern, 0, sizeof(cl_mem), &delta.buf);
  CHK_ERR(err);

  err = clSetKernelArg(kern, 1, sizeof(cl_mem), &actual.buf);
  CHK_ERR(err);

  err = clSetKernelArg(kern, 2, sizeof(cl_mem), &prev_out.buf);
  CHK_ERR(err);

  err = clSetKernelArg(kern, 3, sizeof(cl_mem), &prev_v.buf);
  CHK_ERR(err);

  err = clEnqueueNDRangeKernel(cv.commands[0],
          kern,
          1,
          NULL,
          global_work_size,
          local_work_size,
          0,
          NULL,
          NULL
  );
  CHK_ERR(err);
}


void hl_back_ocl(GpuBatch delta, GpuBatch v, cl_kernel kern, cl_vars_t cv) {
  cl_int err = CL_SUCCESS;
  size_t global_work_size[2] = {static_cast<size_t>(delta.batch_size * delta.rows * delta.cols * delta.depth)};
  size_t local_work_size[2] = {512};

  /* Set kernel arguments */

  err = clSetKernelArg(kern, 0, sizeof(cl_mem), &delta.buf);
  CHK_ERR(err);

  err = clSetKernelArg(kern, 1, sizeof(cl_mem), &v.buf);
  CHK_ERR(err);

  err = clEnqueueNDRangeKernel(cv.commands[0],
          kern,
          1,
          NULL,
          global_work_size,
          local_work_size,
          0,
          NULL,
          NULL
  );
  CHK_ERR(err);
}
