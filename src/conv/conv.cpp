#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cassert>
#include <cmath>

#include "clhelp.h"
//#include "convolution.h"


#include <opencv2/core/core.hpp>

using namespace cv;

void ocl_conv(cl_mem input, int rows, int cols, cl_mem output, float *weights, int r, cl_vars_t cv, cl_kernel conv) {

    cl_mem g_Weights;

    cl_int err = CL_SUCCESS;

    int weights_size = (r * 2 + 1) * (r * 2 + 1) * sizeof(float);
    g_Weights = clCreateBuffer(cv.context, CL_MEM_READ_ONLY,
            weights_size, NULL, &err);
    CHK_ERR(err);

    err = clEnqueueWriteBuffer(cv.commands, g_Weights, true, 0, weights_size,
            weights, 0, NULL, NULL);
    CHK_ERR(err);


    size_t global_work_size[2] = {static_cast<size_t>(rows), static_cast<size_t>(cols)};
    size_t local_work_size[2] = {static_cast<size_t>(rows), static_cast<size_t>(cols)};

    /* Set kernel arguments */

    err = clSetKernelArg(conv, 0, sizeof(cl_mem), &output);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 1, sizeof(cl_mem), &input);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 2, sizeof(cl_mem), &g_Weights);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 3, sizeof(int), &rows);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 4, sizeof(int), &cols);
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
            NULL,
            0,
            NULL,
            NULL
    );
    CHK_ERR(err);


    /* Block until all queued OpenCL commands in command-queue are completed */
    err = clFinish(cv.commands);
    CHK_ERR(err);


    clReleaseMemObject(g_Weights);
}
