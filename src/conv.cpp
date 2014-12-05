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

void ocl_conv(GpuBatch input, GpuBatch output, GpuWeights weights, int r, cl_vars_t cv, cl_kernel conv) {

    cl_int err = CL_SUCCESS;

    size_t global_work_size[3] = {static_cast<size_t>(input.rows), static_cast<size_t>(input.cols), static_cast<size_t>(input.batch_size)};
    size_t local_work_size[3] = {static_cast<size_t>(input.rows), static_cast<size_t>(input.cols), 1};

    /* Set kernel arguments */

    err = clSetKernelArg(conv, 0, sizeof(cl_mem), &output.buf);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 1, sizeof(cl_mem), &input.buf);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 2, sizeof(cl_mem), &weights.buf);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 3, sizeof(int), &input.rows);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 4, sizeof(int), &input.cols);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 5, sizeof(int), &input.depth);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 6, sizeof(int), &r);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 7, sizeof(int), &r);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 8, sizeof(int), &weights.num_sets);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 9, sizeof(float) * input.rows * input.cols * input.depth, NULL);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 10, sizeof(float) * input.rows * input.cols, NULL);
    CHK_ERR(err);

    /* Enqueue a command to execute the matmul kernel on the device associated with the
    * command queue. */
    err = clEnqueueNDRangeKernel(cv.commands,
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


    /* Block until all queued OpenCL commands in command-queue are completed */
    err = clFinish(cv.commands);
    CHK_ERR(err);
}

void cl_max_pool(cl_mem input, int in_rows, int in_cols, GpuBatch output, int s, cl_vars_t cv, cl_kernel conv) {

    cl_int err = CL_SUCCESS;

    size_t global_work_size[3] = {static_cast<size_t>(output.rows), static_cast<size_t>(output.cols), static_cast<size_t>(output.batch_size)};
    size_t local_work_size[3] = {static_cast<size_t>(output.rows), static_cast<size_t>(output.cols), 1};

    /* Set kernel arguments */

    err = clSetKernelArg(conv, 0, sizeof(cl_mem), &output.buf);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 1, sizeof(cl_mem), &input);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 2, sizeof(int), &in_rows);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 3, sizeof(int), &in_cols);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 4, sizeof(int), &s);
    CHK_ERR(err);

    /* Enqueue a command to execute the matmul kernel on the device associated with the
    * command queue. */
    err = clEnqueueNDRangeKernel(cv.commands,
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


    /* Block until all queued OpenCL commands in command-queue are completed */
    err = clFinish(cv.commands);
    CHK_ERR(err);
}
