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

void ocl_conv(GpuMat input, cl_mem output, cl_mem weights, int r, cl_vars_t cv, cl_kernel conv) {

    cl_int err = CL_SUCCESS;

    size_t global_work_size[2] = {static_cast<size_t>(input.rows), static_cast<size_t>(input.cols)};
    size_t local_work_size[2] = {static_cast<size_t>(input.rows), static_cast<size_t>(input.cols)};

    /* Set kernel arguments */

    err = clSetKernelArg(conv, 0, sizeof(cl_mem), &output);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 1, sizeof(cl_mem), &input.buf);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 2, sizeof(cl_mem), &weights);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 3, sizeof(int), &input.rows);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 4, sizeof(int), &input.cols);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 5, sizeof(int), &r);
    CHK_ERR(err);

    /* Enqueue a command to execute the matmul kernel on the device associated with the
    * command queue. */
    err = clEnqueueNDRangeKernel(cv.commands,
            conv,
            2, //2 work dimensions
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

void cl_max_pool(cl_mem input, int in_cols, int rows, int cols, cl_mem output, int s, cl_vars_t cv, cl_kernel conv) {

    cl_int err = CL_SUCCESS;

    size_t global_work_size[2] = {static_cast<size_t>(rows), static_cast<size_t>(cols)};
    size_t local_work_size[2] = {static_cast<size_t>(rows), static_cast<size_t>(cols)};

    /* Set kernel arguments */

    err = clSetKernelArg(conv, 0, sizeof(cl_mem), &output);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 1, sizeof(cl_mem), &input);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 2, sizeof(int), &in_cols);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 3, sizeof(int), &s);
    CHK_ERR(err);

    /* Enqueue a command to execute the matmul kernel on the device associated with the
    * command queue. */
    err = clEnqueueNDRangeKernel(cv.commands,
            conv,
            2, //2 work dimensions
            NULL,
            global_work_size,
            NULL,
            0,
            NULL,
            NULL
    );
    CHK_ERR(err);


    /* Block until all queued OpenCL commands in command-queue are completed */
    err = clFinish(cv.commands);
    CHK_ERR(err);
}
