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

void layer1_ocl(GpuBatch input, GpuBatch output, cl_mem weights, int r, int s, cl_vars_t cv, cl_kernel conv, cl_kernel pool) {
    cl_int err = CL_SUCCESS;
    cl_mem conv_out = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
                                     input.batch_size * (input.rows - 2 * r) * (input.cols - 2 * r)
                                     * sizeof(float), NULL, &err);
    CHK_ERR(err);
    ocl_conv(input, conv_out, weights, r, cv, conv);
    cl_max_pool(conv_out, input.rows - 2 * r, input.cols - 2 * r, output, s, cv, pool);
}

void layer2_ocl(vector<GpuBatch> inputs, GpuBatch output, vector<cl_mem> weights, int r, int s, cl_vars_t cv, cl_kernel conv, cl_kernel pool) {
  int rows = inputs[0].rows;
  int cols = inputs[0].cols;
  cl_int err = CL_SUCCESS;
  cl_mem conv_out = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
                                   inputs[0].batch_size * (rows - 2 * r) * (cols - 2 * r)
                                   * sizeof(float), NULL, &err);
  CHK_ERR(err);
  for (size_t i = 0; i < weights.size(); ++i) {
      ocl_conv(inputs[i], conv_out, weights[i], r, cv, conv);
  }
  cl_max_pool(conv_out, rows - 2 * r, cols - 2 * r, output, s, cv, pool);
}

float *gen_random_weights(int radius, float range) {
    float *weights = new float[(radius * 2 + 1) * (radius * 2 + 1)];
    for (int i = 0; i < (radius * 2 + 1); i++) {
        for (int j = 0; j < (radius * 2 + 1); j++) {
            weights[i * (radius * 2 + 1) + j] = ((float) rand() / (float) RAND_MAX) * (range * 2 + 1) - range;
        }
    }
    return weights;
}

void check_outputs(Batch ocl_out, Batch out) {
  int rows = ocl_out.rows;
  int cols = ocl_out.cols;
  for (int z = 0; z < ocl_out.batch_size; z++) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        if (ocl_out.data[z * rows * cols + i * cols + j] != 
            out.data[z * rows * cols + i * cols + j]) {
          cout << "Not equal at (" << i << ", " << j << ")" << endl;
          cout << "SERIAL: " << out.data[z * rows * cols + i * cols + j] << endl;
          cout << "OCL   : " << ocl_out.data[z * rows * cols + i * cols + j] << endl;
          exit(1);
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
    int batch_size = 10;
    Batch images(batch_size, image.rows, image.cols);
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

    int l1_numoutputs = 4;
    vector<Batch> l1_outputs;
    vector<GpuBatch> l1_ocl_outputs;
    int w = 2;
    float *l1_weights = gen_random_weights(w, 1);
    cl_mem ocl_l1_weights = clCreateBuffer(cv.context, CL_MEM_READ_ONLY,
                                           (w * 2 + 1) * (w * 2 + 1) *
                                           sizeof(float), NULL, &err);
    CHK_ERR(err);
    err = clEnqueueWriteBuffer(cv.commands, ocl_l1_weights, true, 0,
                               (w * 2 + 1) * (w * 2 + 1) *
                               sizeof(float), l1_weights, 0, NULL,
                               NULL);
    CHK_ERR(err);

    // convolution layer and pooling
    for (int i = 0; i < l1_numoutputs; i++) {
        Batch out(batch_size, (image.rows - 2 * w) / 2, (image.cols - 2 * w) / 2);
        double t0 = timestamp();
        layer1_compute(images, out, l1_weights, w, 2);
        t0 = timestamp() - t0;
        cout << "layer1 naive time " << t0 << endl;
        l1_outputs.push_back(out);
    }
        
    for (int i = 0; i < l1_numoutputs; i++) {
        Batch ocl_out = Batch(batch_size, (image.rows - 2 * w) / 2, (image.cols - 2 * w) / 2);
        GpuBatch ocl_out_buf = GpuBatch(ocl_out, cv);
        double t0 = timestamp();
        layer1_ocl(ocl_images, ocl_out_buf,
                   ocl_l1_weights, w, 2, cv, conv, pool);
        t0 = timestamp() - t0;
        cout << "layer1 ocl time " << t0 << endl;
        err = clEnqueueReadBuffer(cv.commands, ocl_out_buf.buf, true, 0,
                                  batch_size * ocl_out.rows * ocl_out.cols * sizeof(float),
                                  ocl_out.data, 0, NULL,
                                  NULL);
        CHK_ERR(err);
        
        l1_ocl_outputs.push_back(ocl_out_buf);
        check_outputs(ocl_out, l1_outputs[i]);
    }

    vector<vector<float *> > l2_weights;
    vector<vector<cl_mem> > ocl_l2_weights;
    int l2_numoutputs = 6;
    for (int i = 0; i < l2_numoutputs; ++i) {
        vector<float *> weights(0);
        vector<cl_mem> ocl_weights(0);
        for (int j = 0; j < l1_numoutputs; ++j) {
            weights.push_back(gen_random_weights(2, l1_numoutputs));
            cl_mem ocl_weight = clCreateBuffer(cv.context, CL_MEM_READ_ONLY,
                                                (w * 2 + 1) * (w * 2 + 1) * sizeof(float), NULL, &err);
            CHK_ERR(err);
            err = clEnqueueWriteBuffer(cv.commands, ocl_weight, true, 0, 
                                       (w * 2 + 1) * (w * 2 + 1) * sizeof(float), 
                                       weights.back(), 0, NULL, NULL);
            CHK_ERR(err);
            ocl_weights.push_back(ocl_weight);
        }
        l2_weights.push_back(weights);
        ocl_l2_weights.push_back(ocl_weights);
    }

    vector<Batch> l2_outputs;
    vector<Batch> ocl_l2_outputs;
    for (vector<float *> weights : l2_weights) {
        double t0 = timestamp();
        Batch out(batch_size, (l1_outputs[0].rows - 2 * 2) / 2, 
            (l1_outputs[0].cols - 2 * 2) / 2);
        layer2_compute(l1_outputs, out, weights, 2, 2);
        t0 = timestamp() - t0;
        cout << "layer2 naive time " << t0 << endl;
        l2_outputs.push_back(out);
    }
    for (vector<cl_mem> weights : ocl_l2_weights) {
        Batch output(l1_ocl_outputs[0].batch_size, 
            (l1_ocl_outputs[0].rows - 2 * 2) / 2, 
            (l1_ocl_outputs[0].cols - 2 * 2) / 2);
        GpuBatch gpu_output(output, cv);
        double t0 = timestamp();
        layer2_ocl(l1_ocl_outputs, gpu_output, weights, 2, 2, cv, conv, pool);
        t0 = timestamp() - t0;
        cout << "layer2 ocl time " << t0 << endl;
        err = clEnqueueReadBuffer(cv.commands, gpu_output.buf, true, 0, 
                                  output.batch_size * output.rows * output.cols
                                  * sizeof(float), output.data, 0, NULL,
                                  NULL);
        CHK_ERR(err);
        err = clFinish(cv.commands);
        CHK_ERR(err);
        ocl_l2_outputs.push_back(output);
    }
    for (size_t i = 0; i < ocl_l2_outputs.size(); i++) {
        check_outputs(ocl_l2_outputs[i], l2_outputs[i]);
    }

    cout << "PASSED" << endl;
    
    return 0;
}
