#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <random>
#include "src/conv.cpp"
#include "src/serial.cpp"
#include <cstdlib>
#include <cstdlib>
#include <clBLAS.h>

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
    if (argc != 2) {
        cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }


    // cout << "img_data = " << endl << " " << image << endl << endl;

    cl_vars_t cv;
    initialize_ocl(cv);

    cl_kernel conv = build_kernel(string("conv"), string("./src/kernels.cl"), cv);
    cl_kernel tanh_kern = build_kernel(string("ocl_tan_h"), string("./src/kernels.cl"), cv);

    cl_int err = CL_SUCCESS;
    Mat image, int_image;
    VideoCapture sequence(argv[1]);
    sequence >> int_image;
    int_image.convertTo(image, CV_32F);
    int batch_size = 512;
    vector<Batch> batches;
    vector<GpuBatch> gpu_batches;
    for (cl_uint device = 0; device < cv.num_devices; device++) {
      Batch images(batch_size, 1, image.rows, image.cols);
      for (int z = 0; z < batch_size; z++) {
        sequence >> int_image;
        int_image.convertTo(image, CV_32F);
        memcpy(&images.data[z * image.rows * image.cols], (float*)image.data, image.total() * image.elemSize());
      }

      if (!image.data) {
          cout << "Could not open or find the image" << std::endl;
          return -1;
      }
      batches.push_back(images);
      GpuBatch ocl_images = GpuBatch(images, cv, device);
      gpu_batches.push_back(ocl_images);
    }


    int l1_numoutputs = 20;
    int w = 2;
    int l2_numoutputs = 52;
    vector<Weights> l1_weights;
    vector<float> l1_bias;
    vector<Weights> l2_weights;
    vector<float> l2_bias;
    for (int i = 0; i < l1_numoutputs; i++) {
      l1_weights.push_back(Weights(w, 1, 1 + l1_numoutputs));
      l1_bias.push_back(0);
    }
    for (int i = 0; i < l2_numoutputs; i++) {
      l2_weights.push_back(Weights(w, l1_numoutputs, l2_numoutputs + l1_numoutputs));
      l2_bias.push_back(0);
    }

    vector<vector<GpuWeights> > l1_gpu_weights;
    vector<vector<GpuWeights> > l2_gpu_weights;
    for (cl_uint device = 0; device < cv.num_devices; device++) {
      vector<GpuWeights> l1_weights_set;
      for (int i = 0; i < l1_numoutputs; i++) {
        l1_weights_set.push_back(GpuWeights(l1_weights[i], cv, device));
      }
      vector<GpuWeights> l2_weights_set;
      for (int i = 0; i < l2_numoutputs; i++) {
        l2_weights_set.push_back(GpuWeights(l2_weights[i], cv, device));
      }
      l1_gpu_weights.push_back(l1_weights_set);
      l2_gpu_weights.push_back(l2_weights_set);
    }

    vector<Batch> cpu_l1_outputs;
    vector<Batch> cpu_l2_outputs;
    for (cl_uint device = 0; device < cv.num_devices; device++) {
      Batch l1_output(batch_size, l1_numoutputs, (image.rows - 2 * w) / 2, 
               (image.cols - 2 * w) / 2);
      cpu_l1_outputs.push_back(l1_output);
      Batch l2_out(batch_size, l2_numoutputs, (l1_output.rows - 2 * 2) / 2, 
          (l1_output.cols - 2 * 2) / 2);
      cpu_l2_outputs.push_back(l2_out);
    }
    double cpu_total = timestamp();
    for (cl_uint device = 0; device < cv.num_devices; device++) {
      Batch batch = batches[device];
      layer1_compute(batch, cpu_l1_outputs[device], l1_weights, w, 2, l1_bias);
      layer1_compute(cpu_l1_outputs[device], cpu_l2_outputs[device], l2_weights, 2, 2, l2_bias);
    }
    cpu_total = timestamp() - cpu_total;
    vector<GpuBatch> gpu_l1_outputs;
    vector<GpuBatch> gpu_l2_outputs;
    vector<Batch> gpu_l1_outputs_host;
    vector<Batch> gpu_l2_outputs_host;
    for (cl_uint device = 0; device < cv.num_devices; device++) {
      Batch ocl_out = Batch(batch_size, l1_numoutputs,
          (image.rows - 2 * w) / 2, (image.cols - 2 * w) / 2);
      GpuBatch ocl_out_buf = GpuBatch(ocl_out, cv, device);
      gpu_l1_outputs.push_back(ocl_out_buf);
      gpu_l1_outputs_host.push_back(ocl_out);
      Batch l2_ocl_out = Batch(batch_size, l2_numoutputs,
          (ocl_out.rows - 2 * w) / 2, (ocl_out.cols - 2 * w) / 2);
      GpuBatch l2_ocl_out_buf = GpuBatch(l2_ocl_out, cv, device);
      gpu_l2_outputs.push_back(l2_ocl_out_buf);
      gpu_l2_outputs_host.push_back(l2_ocl_out);
    }
    double gpu_total = timestamp();
    for (cl_uint device = 0; device < cv.num_devices; device++) {
      GpuBatch batch = gpu_batches[device];
      GpuBatch ocl_out_buf = gpu_l1_outputs[device];
      ocl_conv(batch, ocl_out_buf, l1_gpu_weights[device], l1_bias, w, cv, conv, device);
      err = clFinish(cv.commands[device]);
      ocl_conv(ocl_out_buf, gpu_l2_outputs[device], l2_gpu_weights[device], l2_bias, w, cv, conv, device);
    }
    for (cl_uint device = 0; device < cv.num_devices; device++) {
      err = clFinish(cv.commands[device]);
      CHK_ERR(err);
    }
    gpu_total = timestamp() - gpu_total;
    for (cl_uint device = 0; device < cv.num_devices; device++) {
      GpuBatch out_buf = gpu_l1_outputs[device];
      Batch out = gpu_l1_outputs_host[device];
      err = clEnqueueReadBuffer(cv.commands[device], out_buf.buf, true, 0,
                                batch_size * out_buf.rows * out_buf.cols *
                                out_buf.depth * sizeof(float),
                                out.data, 0, NULL, NULL);
      CHK_ERR(err);
      err = clFinish(cv.commands[device]);
      CHK_ERR(err);

      check_outputs(out, cpu_l1_outputs[device]);
      out_buf = gpu_l2_outputs[device];
      out = gpu_l2_outputs_host[device];
      err = clEnqueueReadBuffer(cv.commands[device], out_buf.buf, true, 0,
                                batch_size * out_buf.rows * out_buf.cols *
                                out_buf.depth * sizeof(float),
                                out.data, 0, NULL, NULL);
      CHK_ERR(err);
      err = clFinish(cv.commands[device]);
      CHK_ERR(err);
      check_outputs(out, cpu_l2_outputs[device]);
    }
    err = clblasSetup();
    if (err != CL_SUCCESS) {
      printf("clblasSetup() failed with %d\n", err);
      return 1;
    }
    // hidden layer
    vector<GpuBatch> hl1_out;
    vector<GpuBatch> hl1_v;
    for (cl_uint device = 0; device < 1; device++) {
      GpuBatch batch = gpu_l2_outputs[device];
      Weights hidden_layer_weights(0, batch.rows * batch.cols * batch.depth * 
          batch.rows * batch.cols * batch.depth, batch.rows * batch.cols * batch.depth);
      GpuWeights weights(hidden_layer_weights, cv, device);
      Weights bias(0, batch.rows * batch.cols * batch.depth, 0);
      GpuWeights bias_buf(bias, cv, device);
      Batch out = Batch(batch.batch_size, batch.rows, batch.cols, batch.depth);
      GpuBatch out_buf = GpuBatch(out, cv, device);
      Batch v = Batch(batch.batch_size, batch.rows, batch.cols, batch.depth);
      GpuBatch v_buf = GpuBatch(v, cv, device);
      for (int offset = 0; offset < batch.batch_size; offset++) {
        err = clblasSgemv(clblasRowMajor, clblasNoTrans, 
                          batch.rows * batch.cols * batch.depth, 
                          batch.rows * batch.cols * batch.depth, 
                          1.0f, weights.buf, 
                          0, batch.rows * batch.cols * batch.depth,
                          batch.buf, 
                          offset * batch.rows * batch.cols * batch.depth, 1,
                          1.0f, v_buf.buf, 
                          offset * batch.rows * batch.cols * batch.depth, 1, 1,
                          &cv.commands[device], 0, NULL, NULL);
        CHK_ERR(err);
        err = clblasSaxpy(batch.rows * batch.cols * batch.depth, 1.0f, 
                          bias_buf.buf, 0, 1,
                          v_buf.buf,
                          offset * batch.rows * batch.cols * batch.depth, 1,
                          1, &cv.commands[device], 0, NULL, NULL);
        CHK_ERR(err);
      }
      ocl_tanh(out_buf, v_buf, cv, tanh_kern, device);
      hl1_out.push_back(out_buf);
      hl1_v.push_back(v_buf);
    }
    vector<GpuBatch> hl2_out;
    vector<GpuBatch> hl2_v;
    for (cl_uint device = 0; device < 1; device++) {
      GpuBatch batch = hl1_out[device];
      Weights hidden_layer_weights(0, batch.rows * batch.cols * batch.depth * 2,
          batch.rows * batch.cols * batch.depth);
      GpuWeights weights(hidden_layer_weights, cv, device);
      Batch out = Batch(batch.batch_size, 1, 1, 2);
      GpuBatch out_buf = GpuBatch(out, cv, device);
      Batch v = Batch(batch.batch_size, 1, 1, 2);
      GpuBatch v_buf = GpuBatch(v, cv, device);
      for (int offset = 0; offset < batch.batch_size; offset++) {
        err = clblasSgemv(clblasRowMajor, clblasNoTrans, 
                          2,
                          batch.rows * batch.cols * batch.depth, 
                          1.0f, weights.buf, 
                          0, batch.rows * batch.cols * batch.depth,
                          batch.buf, 
                          offset * batch.rows * batch.cols * batch.depth, 1,
                          1.0f, v_buf.buf, 
                          offset * 2, 1, 1,
                          &cv.commands[device], 0, NULL, NULL);
        CHK_ERR(err);
      }
      ocl_tanh(out_buf, v_buf, cv, tanh_kern, device);
      hl2_out.push_back(out_buf);
      hl2_v.push_back(v_buf);
      clFinish(cv.commands[device]);
    }
    for (cl_uint device = 0; device < 1; device++) {
      GpuBatch out_buf = hl2_out[device];
      float* out = new float[batch_size * out_buf.rows * out_buf.cols * out_buf.depth];
      err = clEnqueueReadBuffer(cv.commands[device], out_buf.buf, true, 0,
                                batch_size * out_buf.rows * out_buf.cols *
                                out_buf.depth * sizeof(float),
                                out, 0, NULL, NULL);
      CHK_ERR(err);
      err = clFinish(cv.commands[device]);
      CHK_ERR(err);
      for (int i = 0; i < out_buf.rows; i++) {
        for (int j = 0; j < out_buf.cols; j++ ) {
          cout << out[i * out_buf.cols + j] << " ";
        }
        cout << endl;
      }
    }

    cout << "PASSED" << endl;
    cout << "cpu time: " << cpu_total << endl;
    cout << "gpu time: " << gpu_total << endl;
    cout << "gpu speedup: " << cpu_total / gpu_total << endl;
    return 0;
}
