#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <random>
#include "src/conv.cpp"
#include "src/serial.cpp"
#include <cstdlib>
#include <cstdlib>

using namespace cv;
using namespace std;

void layer1_ocl(GpuBatch input, GpuBatch output, GpuWeights weights,
                int r, int s, cl_vars_t cv, cl_kernel conv) {
    ocl_conv(input, output, weights, r, cv, conv);
}

void check_outputs(Batch ocl_out, Batch out) {
  int rows = ocl_out.rows;
  int cols = ocl_out.cols;
  int depth = ocl_out.depth;
  for (int z = 0; z < ocl_out.batch_size; z++) {
    for (int d = 0; d < ocl_out.depth; d++) {
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          int idx = z * rows * cols * depth + d * rows * cols + i * cols + j;
          if (ocl_out.data[idx] != out.data[idx]) {
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
    if (argc != 2) {
        cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }


    // cout << "img_data = " << endl << " " << image << endl << endl;

    cl_vars_t cv;
    initialize_ocl(cv);

    cl_kernel conv = build_kernel(string("conv"), string("./src/kernels.cl"), cv);

    cl_int err = CL_SUCCESS;
    Mat image, int_image;
    VideoCapture sequence(argv[1]);
    sequence >> int_image;
    int_image.convertTo(image, CV_32F);
    int batch_size = 500;
    Batch images(batch_size, 1, image.rows, image.cols);
    for (int z = 0; z < batch_size; z++) {
      sequence >> int_image;
      int_image.convertTo(image, CV_32F);
      memcpy(&images.data[z], (float*)image.data, image.total() * image.elemSize());
    }

    if (!image.data) {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    GpuBatch ocl_images = GpuBatch(images, cv);

    int l1_numoutputs = 20;
    int w = 2;
    Weights l1_weights(l1_numoutputs, w, 1, 1);
    GpuWeights l1_ocl_weights(l1_weights, cv);

    // convolution layer and pooling
    Batch l1_output(batch_size, l1_numoutputs, (image.rows - 2 * w) / 2, 
             (image.cols - 2 * w) / 2);
    double l1_t = timestamp();
    layer1_compute(images, l1_output, l1_weights, w, 2);
    l1_t = timestamp() - l1_t;
        
    Batch ocl_out = Batch(batch_size, l1_numoutputs,
        (image.rows - 2 * w) / 2, (image.cols - 2 * w) / 2);
    GpuBatch ocl_out_buf = GpuBatch(ocl_out, cv);
    double l1_ocl_t = timestamp();
    layer1_ocl(ocl_images, ocl_out_buf,
               l1_ocl_weights, w, 2, cv, conv);
    l1_ocl_t = timestamp() - l1_ocl_t;
    err = clEnqueueReadBuffer(cv.commands, ocl_out_buf.buf, true, 0,
                              batch_size * ocl_out.rows * ocl_out.cols *
                              ocl_out.depth * sizeof(float),
                              ocl_out.data, 0, NULL,
                              NULL);
    CHK_ERR(err);
    err = clFinish(cv.commands);
    CHK_ERR(err);
    
    check_outputs(ocl_out, l1_output);
    cout << "layer1 naive time " << l1_t << endl;
    cout << "layer1 ocl time " << l1_ocl_t << endl;
    clReleaseMemObject(ocl_images.buf);


    int l2_numoutputs = 50;
    Weights l2_weights(l2_numoutputs, w, l1_numoutputs, l1_numoutputs);
    GpuWeights l2_ocl_weights(l2_weights, cv);

    Batch l2_out(batch_size, l2_numoutputs, (l1_output.rows - 2 * 2) / 2, 
        (l1_output.cols - 2 * 2) / 2);
    double l2_t = timestamp();
    layer1_compute(l1_output, l2_out, l2_weights, 2, 2);
    l2_t = timestamp() - l2_t;

    Batch l2_ocl_out = Batch(batch_size, l2_numoutputs,
        (l1_output.rows - 2 * w) / 2, (l1_output.cols - 2 * w) / 2);
    GpuBatch l2_ocl_out_buf = GpuBatch(ocl_out, cv);
    double l2_ocl_t = timestamp();
    layer1_ocl(ocl_out_buf, l2_ocl_out_buf,
               l2_ocl_weights, w, 2, cv, conv);
    l2_ocl_t = timestamp() - l2_ocl_t;
    err = clEnqueueReadBuffer(cv.commands, l2_ocl_out_buf.buf, true, 0,
                              batch_size * l2_ocl_out.rows * l2_ocl_out.cols *
                              l2_ocl_out.depth * sizeof(float),
                              l2_ocl_out.data, 0, NULL,
                              NULL);
    CHK_ERR(err);
    err = clFinish(cv.commands);
    CHK_ERR(err);
    check_outputs(l2_ocl_out, l2_out);
    cout << "layer2 naive time " << l2_t << endl;
    cout << "layer2 ocl time " << l2_ocl_t << endl;

    cout << "PASSED" << endl;
    
    return 0;
}
