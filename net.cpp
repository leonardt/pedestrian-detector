#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <random>
#include "src/layer.cpp"
#include "src/types.h"
#include <cstdlib>

using namespace cv;
using namespace std;

void check_outputs(Batch ocl_out, Batch out) {
  int rows = ocl_out.rows;
  int cols = ocl_out.cols;
  int depth = ocl_out.depth;
  for (int z = 0; z < ocl_out.batch_size; z++) {
    for (int d = 0; d < ocl_out.depth; d++) {
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          int idx = z * rows * cols * depth + d * rows * cols + i * cols + j;
          if (ocl_out.data[idx] - out.data[idx] > 1) {
            cout << "Not equal at (" << d << ", " << i << ", " << j << ")" << endl;
            cout << "SERIAL: " << out.data[idx] << endl;
            cout << "OCL   : " << ocl_out.data[idx] << endl;
            exit(1);
          }
        }
      }
    }
  }
}

int main(int argc, char **argv) {
    srand(time(NULL));
    if (argc < 2) {
        cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }


    // cout << "img_data = " << endl << " " << image << endl << endl;

    cl_vars_t cv;
    initialize_ocl(cv);

    cl_kernel conv = build_kernel(string("conv"), string("./src/kernels.cl"), cv);
    cl_kernel tanh_kern = build_kernel(string("ocl_tan_h"), string("./src/kernels.cl"), cv);
    cl_kernel ol_back_kern = build_kernel(string("ol_back"), string("./src/kernels.cl"), cv);
    cl_kernel hl_back_kern = build_kernel(string("hl_back"), string("./src/kernels.cl"), cv);


    int l1_numoutputs = 20;
    int w = 2;
    int l2_numoutputs = 52;

    cl_int err = CL_SUCCESS;
    Mat image, int_image;
    VideoCapture peds(argv[1]);
    VideoCapture non_peds(argv[2]);
    peds >> int_image;
    int_image.convertTo(image, CV_32F);
    int batch_size = 512;
    vector<Batch*> batches;
    vector<GpuBatch*> actuals;
    vector<GpuBatch*> gpu_batches;
    vector<ConvPoolLayer*> cpu_l1s;
    vector<ConvPoolLayer*> cpu_l2s;
    vector<GpuConvPoolLayer*> gpu_l1s;
    vector<GpuConvPoolLayer*> gpu_l2s;
    for (cl_uint device = 0; device < cv.num_devices; device++) {
      Batch* images = new Batch(batch_size, 1, image.rows, image.cols);
      Batch actual(batch_size, 1, 1, 2);
      for (int z = 0; z < batch_size - 1; z++) {
        float r = (float) rand() / (float) RAND_MAX;
        if (r > .5) {
          peds >> int_image;
          actual.data[z * 2] = 1;
          actual.data[z * 2 + 1] = -1;
        } else {
          non_peds >> int_image;
          actual.data[z * 2] = -1;
          actual.data[z * 2 + 1] = 1;
        }
        int_image.convertTo(image, CV_32F);
        if (!image.data) {
            cout << "Could not open or find the image" << std::endl;
            return -1;
        }
        memcpy(&images->data[z * image.rows * image.cols], (float*)image.data, image.total() * image.elemSize());
      }

      batches.push_back(images);
      GpuBatch* ocl_images = new GpuBatch(*images, cv, device);
      GpuBatch* ocl_actual = new GpuBatch(actual, cv, device);
      actuals.push_back(ocl_actual);
      gpu_batches.push_back(ocl_images);

      ConvPoolLayer* cpu_l1 = new ConvPoolLayer(batch_size, image.rows, image.cols, 1, w, 
          l1_numoutputs);
      ConvPoolLayer* cpu_l2 = new ConvPoolLayer(batch_size, (image.rows - 2 * w) / 2, 
                           (image.cols - 2 * w) / 2, l1_numoutputs, w, 
                           l2_numoutputs);

      GpuConvPoolLayer* gpu_l1 = new GpuConvPoolLayer(*cpu_l1, cv, conv, device);
      GpuConvPoolLayer* gpu_l2 = new GpuConvPoolLayer(*cpu_l2, cv, conv, device);
      cpu_l1s.push_back(cpu_l1);
      cpu_l2s.push_back(cpu_l2);
      gpu_l1s.push_back(gpu_l1);
      gpu_l2s.push_back(gpu_l2);
    }


    double cpu_total = timestamp();
    for (cl_uint device = 0; device < cv.num_devices; device++) {
      Batch batch = *batches[device];
      cpu_l1s[device]->forward(&batch);
      cpu_l2s[device]->forward(cpu_l1s[device]->output);
    }
    cpu_total = timestamp() - cpu_total;

    double gpu_total = timestamp();
    for (cl_uint device = 0; device < cv.num_devices; device++) {
      gpu_l1s[device]->forward(gpu_batches[device]);
      gpu_l2s[device]->forward(gpu_l1s[device]->output);
    }
    for (cl_uint device = 0; device < cv.num_devices; device++) {
      err = clFinish(cv.commands[device]);
    }
    gpu_total = timestamp() - gpu_total;
    for (cl_uint device = 0; device < cv.num_devices; device++) {
      GpuBatch out_buf = *gpu_l1s[device]->output;
      Batch out(out_buf.batch_size, out_buf.depth, out_buf.rows, out_buf.cols);
      err = clEnqueueReadBuffer(cv.commands[device], out_buf.buf, true, 0,
                                out_buf.batch_size * out_buf.rows * out_buf.cols *
                                out_buf.depth * sizeof(float),
                                out.data, 0, NULL, NULL);
      CHK_ERR(err);
      err = clFinish(cv.commands[device]);
      CHK_ERR(err);

      check_outputs(out, *cpu_l1s[device]->output);
      GpuBatch l2_out_buf = *gpu_l2s[device]->output;
      Batch l2_out(l2_out_buf.batch_size, l2_out_buf.depth, l2_out_buf.rows, l2_out_buf.cols);
      err = clEnqueueReadBuffer(cv.commands[device], l2_out_buf.buf, true, 0,
                                l2_out_buf.batch_size * l2_out_buf.rows * l2_out_buf.cols *
                                l2_out_buf.depth * sizeof(float),
                                l2_out.data, 0, NULL, NULL);
      CHK_ERR(err);
      err = clFinish(cv.commands[device]);
      CHK_ERR(err);
      check_outputs(l2_out, *cpu_l2s[device]->output);
    }

    cout << "PASSED" << endl;
    cout << "cpu time: " << cpu_total << endl;
    cout << "gpu time: " << gpu_total << endl;
    cout << "gpu speedup: " << cpu_total / gpu_total << endl;
    exit(0);

    err = clblasSetup();
    if (err != CL_SUCCESS) {
      printf("clblasSetup() failed with %d\n", err);
      return 1;
    }
    // hidden layer
    GpuBatch l2_output = *gpu_l2s[0]->output;
    HiddenLayer hl1(l2_output.rows * l2_output.cols * l2_output.depth,
                    l2_output.rows * l2_output.cols * l2_output.depth,
                    l2_output.batch_size, cv, tanh_kern, hl_back_kern);
    hl1.forward(gpu_l2s[0]->output);
    OutputLayer ol(l2_output.rows * l2_output.cols * l2_output.depth,
        2, batch_size, cv, tanh_kern, ol_back_kern, hl_back_kern);
    ol.forward(hl1.output);
    {
      GpuBatch out_buf = *ol.output;
      float* out = new float[batch_size * out_buf.rows * out_buf.cols * out_buf.depth];
      err = clEnqueueReadBuffer(cv.commands[0], out_buf.buf, true, 0,
                                batch_size * out_buf.rows * out_buf.cols *
                                out_buf.depth * sizeof(float),
                                out, 0, NULL, NULL);
      CHK_ERR(err);
      err = clFinish(cv.commands[0]);
      CHK_ERR(err);
      for (int i = 0; i < out_buf.rows; i++) {
        for (int j = 0; j < out_buf.cols; j++ ) {
          cout << out[i * out_buf.cols + j] << " ";
        }
        cout << endl;
      }
    }
    ol.backward(actuals[0], hl1.delta_bias);
    hl1.backward(gpu_l2s[0]->delta_bias);
    return 0;
}
