#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <random>
#include "src/conv.cpp"
#include "src/serial.cpp"
// #include "src/types.cpp"
#include <cstdlib>
#include <cstdlib>

using namespace cv;
using namespace std;

void layer1_ocl(GpuBatch input, GpuBatch output, GpuWeights weights, int r, int s, cl_vars_t cv, cl_kernel conv, cl_kernel pool) {
    /* cl_int err = CL_SUCCESS; */
    /* cl_mem conv_out = clCreateBuffer(cv.context, CL_MEM_READ_WRITE, */
    /*                                  input.batch_size * (input.rows - 2 * r) * (input.cols - 2 * r) */
    /*                                  * sizeof(float), NULL, &err); */
    /* CHK_ERR(err); */
    ocl_conv(input, output, weights, r, cv, conv);
    // cl_max_pool(conv_out, input.rows - 2 * r, input.cols - 2 * r, output, s, cv, pool);
}

/* void layer2_ocl(vector<GpuBatch> inputs, GpuBatch output, vector<cl_mem> weights, int r, int s, cl_vars_t cv, cl_kernel conv, cl_kernel pool) { */
/*   int rows = inputs[0].rows; */
/*   int cols = inputs[0].cols; */
/*   cl_int err = CL_SUCCESS; */
/*   cl_mem conv_out = clCreateBuffer(cv.context, CL_MEM_READ_WRITE, */
/*                                    inputs[0].batch_size * (rows - 2 * r) * (cols - 2 * r) */
/*                                    * sizeof(float), NULL, &err); */
/*   CHK_ERR(err); */
/*   for (size_t i = 0; i < weights.size(); ++i) { */
/*       ocl_conv(inputs[i], conv_out, weights[i], r, cv, conv); */
/*   } */
/*   cl_max_pool(conv_out, rows - 2 * r, cols - 2 * r, output, s, cv, pool); */
/* } */

float *gen_random_weights(int radius, int num, float range) {
    float *weights = new float[num * (radius * 2 + 1) * (radius * 2 + 1)];
    for (int z = 0; z < num; z++) {
      for (int i = 0; i < (radius * 2 + 1); i++) {
          for (int j = 0; j < (radius * 2 + 1); j++) {
              weights[i * (radius * 2 + 1) + j] = ((float) rand() / (float) RAND_MAX) * (range * 2 + 1) - range;
          }
      }
    }
    return weights;
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
    cl_kernel pool = build_kernel(string("max_pool"), string("./src/kernels.cl"), cv);

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
    /* vector<Batch> l1_outputs; */
    /* vector<GpuBatch> l1_ocl_outputs; */
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
               l1_ocl_weights, w, 2, cv, conv, pool);
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

    /* vector<vector<cl_mem> > ocl_l2_weights; */
    /* int l2_numoutputs = 20; */
    /* Weights l2_weights(2, l2_numoutputs, l1_numoutputs); */
    /*     vector<float *> weights(0); */
    /*     vector<cl_mem> ocl_weights(0); */
    /*     for (int j = 0; j < l1_numoutputs; ++j) { */
    /*         weights.push_back(gen_random_weights(2, l1_numoutputs)); */
    /*         cl_mem ocl_weight = clCreateBuffer(cv.context, CL_MEM_READ_ONLY, */
    /*                                             (w * 2 + 1) * (w * 2 + 1) * sizeof(float), NULL, &err); */
    /*         CHK_ERR(err); */
    /*         err = clEnqueueWriteBuffer(cv.commands, ocl_weight, true, 0, */ 
    /*                                    (w * 2 + 1) * (w * 2 + 1) * sizeof(float), */ 
    /*                                    weights.back(), 0, NULL, NULL); */
    /*         CHK_ERR(err); */
    /*         ocl_weights.push_back(ocl_weight); */
    /*     } */
    /*     l2_weights.push_back(weights); */
    /*     ocl_l2_weights.push_back(ocl_weights); */

    /* vector<Batch> l2_outputs; */
    /* vector<Batch> ocl_l2_outputs; */
    /* double l2_t = 0; */
    /* double l2_ocl_t = 0; */
    /* t0 = timestamp(); */
    /* Batch l2_out(batch_size, (l1_output.rows - 2 * 2) / 2, */ 
    /*     (l1_output.cols - 2 * 2) / 2, l2_weights.depth); */
    /* layer2_compute(l1_output, l2_out, l2_weights, 2, 2); */
    /* l2_t += timestamp() - t0; */
    /* for (vector<cl_mem> weights : ocl_l2_weights) { */
    /*     Batch output(l1_ocl_outputs[0].batch_size, */ 
    /*         (l1_ocl_outputs[0].rows - 2 * 2) / 2, */ 
    /*         (l1_ocl_outputs[0].cols - 2 * 2) / 2); */
    /*     GpuBatch gpu_output(output, cv); */
    /*     double t0 = timestamp(); */
    /*     layer2_ocl(l1_ocl_outputs, gpu_output, weights, 2, 2, cv, conv, pool); */
    /*     l2_ocl_t += timestamp() - t0; */
    /*     err = clEnqueueReadBuffer(cv.commands, gpu_output.buf, true, 0, */ 
    /*                               output.batch_size * output.rows * output.cols */
    /*                               * sizeof(float), output.data, 0, NULL, */
    /*                               NULL); */
    /*     CHK_ERR(err); */
    /*     err = clFinish(cv.commands); */
    /*     CHK_ERR(err); */
    /*     ocl_l2_outputs.push_back(output); */
    /* } */
    /* for (size_t i = 0; i < ocl_l2_outputs.size(); i++) { */
    /*     check_outputs(ocl_l2_outputs[i], l2_outputs[i]); */
    /* } */
    /* cout << "layer2 naive time " << l2_t << endl; */
    /* cout << "layer2 ocl time " << l2_ocl_t << endl; */

    cout << "PASSED" << endl;
    
    return 0;
}
