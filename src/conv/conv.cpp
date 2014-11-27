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

void ocl_conv(Mat input, Mat output, float *weights, int r) {
    std::string conv_kernel_str;

    std::string conv_name_str =
            std::string("conv");
    std::string conv_kernel_file =
            std::string("./src/conv/conv.cl");

    cl_vars_t cv;
    cl_kernel conv;

    readFile(conv_kernel_file,
            conv_kernel_str);

    initialize_ocl(cv);

    compile_ocl_program(conv, cv, conv_kernel_str.c_str(),
            conv_name_str.c_str());

//  float *h_Input, *h_Weights, *h_Output, *h_OutputSerial;
    cl_mem g_Input, g_Weights, g_Output;

//  int n = 32;
//  int r = 2;

//  h_Input = new float[n*n];
//  assert(h_Input);
//  h_Weights = new float[n*n];
//  assert(h_Weights);
//  h_Output = new float[n*n];
//  assert(h_Output);
//  h_OutputSerial = new float[n*n];
//  assert(h_OutputSerial);
//  bzero(h_Output, sizeof(float)*n*n);
//  bzero(h_OutputSerial, sizeof(float)*n*n);

//  for(int i = 0; i < (n*n); i++)
//    {
//      h_Input[i] = (float)drand48();
//      h_Weights[i] = 1;
//    }



    cl_int err = CL_SUCCESS;

    /* Allocate Buffers on the GPU. */
    g_Output = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
            output.rows * output.cols * sizeof(float), NULL, &err);
    CHK_ERR(err);

    g_Input = clCreateBuffer(cv.context, CL_MEM_READ_ONLY,
            input.rows * input.cols * sizeof(float), NULL, &err);
    CHK_ERR(err);

    int weights_size = (r * 2 + 1) * (r * 2 + 1) * sizeof(float);
    g_Weights = clCreateBuffer(cv.context, CL_MEM_READ_ONLY,
            weights_size, NULL, &err);
    CHK_ERR(err);

    /* Copy data from host CPU to GPU */

    /* Enqueue commands in order to perform blocking write to the arrays on GPU.
     * Essentially moving data from CPU buffers to GPU buffers */
    err = clEnqueueWriteBuffer(cv.commands, g_Output, true, 0, output.rows * output.cols * sizeof(float),
            (float*)output.data, 0, NULL, NULL);
    CHK_ERR(err);

    err = clEnqueueWriteBuffer(cv.commands, g_Input, true, 0, input.rows * input.cols * sizeof(float),
            (float*)input.data, 0, NULL, NULL);
    CHK_ERR(err);

    err = clEnqueueWriteBuffer(cv.commands, g_Weights, true, 0, weights_size,
            weights, 0, NULL, NULL);
    CHK_ERR(err);


    size_t global_work_size[2] = {static_cast<size_t>(input.rows), static_cast<size_t>(input.cols)};
    size_t local_work_size[2] = {1, 1};

    /* Set kernel arguments */

    err = clSetKernelArg(conv, 0, sizeof(cl_mem), &g_Output);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 1, sizeof(cl_mem), &g_Input);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 2, sizeof(cl_mem), &g_Weights);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 3, sizeof(int), &input.rows);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 4, sizeof(int), &input.cols);
    CHK_ERR(err);

    err = clSetKernelArg(conv, 5, sizeof(int), &r);
    CHK_ERR(err);




    // timing openCL version
//    double t0 = timestamp();
    /* CS194: Launch matrix multiply kernel */



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


    /* Read result of GPU on host CPU */
    err = clEnqueueReadBuffer(cv.commands, g_Output, true, 0, output.rows * output.cols * sizeof(float),
            (float*)output.data, 0, NULL, NULL);
    CHK_ERR(err);
    err = clFinish(cv.commands);
    CHK_ERR(err);

//    t0 = timestamp() - t0;



    //timing serial version

//    double t1 = timestamp();
//    convolve(h_Input, h_OutputSerial, h_Weights, n, r);
//    t1 = timestamp() - t1;

    // for (int i=0; i < n; i++) {
    //   for (int j=0; j < n; j++) {
    //     printf("%0.f ", h_OutputSerial[i*n + j]);
    //   }
    //   printf("\n");
    // }

    // printf("\n GPU here \n");

    // for (int i=0; i < n; i++) {
    //   for (int j=0; j < n; j++) {
    //     printf("%0.f ", h_Output[i*n +j]);
    //   }
    //   printf("\n");
    // }


//    for (int i = 0; i < (n * n); i++) {
//        // printf("%d\n", n);
//        // printf("CPU @ %d: %f\n", i, h_OutputSerial[i]);
//        // printf("GPU @ %d: %f\n", i, h_Output[i]);
//        double d = h_OutputSerial[i] - h_Output[i];
//        d *= d;
//        if (d > 0.0001) {
//            printf("CPU and GPU results do not match!\n");
//            break;
//        }
//    }
    uninitialize_ocl(cv);

//    delete[] h_Input;
//    delete[] h_Weights;
//    delete[] h_Output;
//    delete[] h_OutputSerial;

    clReleaseMemObject(g_Input);
    clReleaseMemObject(g_Weights);
    clReleaseMemObject(g_Output);


    // flops computation is probably very wrong
//    double gpu_flops_s = (2.0 * pow((double) n, 2.0)) / t0;
//    printf("GPU: %g gflops/sec\n", gpu_flops_s / (1e9));
//
//    double cpu_flops_s = (2.0 * pow((double) n, 2.0)) / t1;
//    printf("CPU: %g gflops/sec\n", cpu_flops_s / (1e9));
//    return 0;
}
