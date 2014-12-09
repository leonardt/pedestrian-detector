#include <cstdlib>
#include "conv.cpp"
#include "serial.cpp"
#include "types.h"
#include <clBLAS.h>

using namespace std;

class ConvPoolLayer {
  public:
    vector<Weights> weights;
    vector<float> bias;
    int num_outputs;
    int num_inputs;
    int w;
    Batch* output;

    ConvPoolLayer(int batch_size, int rows, int cols, int n_in, int weights_r, 
          int n_out) : num_inputs(n_in), num_outputs(n_out), w(weights_r) {
      for (int i = 0; i < n_out; i++) {
        weights.push_back(Weights(weights_r, n_in, n_in + n_out));
        bias.push_back(0.0f);
      };
      output = new Batch(batch_size, n_out, (rows - 2 * w) / 2, (cols - 2 * w) / 2);
    }

    void forward(Batch* batch) {
      layer1_compute(*batch, *output, weights, w, 2, bias);
    }
};

class GpuConvPoolLayer {
  public:
    vector<GpuWeights> weights;
    vector<float> bias;
    int num_outputs;
    int num_inputs;
    int w;
    cl_vars_t cv;
    cl_kernel conv;
    GpuBatch* output;
    GpuBatch* delta_bias;
    cl_uint device;

    GpuConvPoolLayer(ConvPoolLayer layer, cl_vars_t cvs, cl_kernel kernel, cl_uint dev): cv(cvs), conv(kernel) {
      num_outputs = layer.num_outputs;
      num_inputs = layer.num_inputs;
      w = layer.w;
      device = dev;
      for (Weights w : layer.weights) {
        weights.push_back(GpuWeights(w, cv, device));
        bias.push_back(0);
      }
      output = new GpuBatch(*layer.output, cv, device);
      Batch delta(layer.output->batch_size, layer.output->depth, 
          layer.output->rows, layer.output->cols);
      delta_bias = new GpuBatch(delta, cv, device);
    }

    void forward(GpuBatch* batch) {
      ocl_conv(*batch, *output, weights, bias, w, cv, conv, device);
    }
};

class HiddenLayer {
  public:
    GpuWeights* weights;
    GpuWeights* bias;
    GpuBatch* output;
    GpuBatch* v;
    cl_vars_t cv;
    int batch_size;
    int num_in;
    int num_out;
    cl_kernel tanh_kern;
    cl_kernel hl_back_kern;
    GpuBatch* delta_bias;

    HiddenLayer(int n_in, int n_out, int b_size, cl_vars_t cvs, cl_kernel tanh,
        cl_kernel hl_back): num_in(n_in), num_out(n_out), cv(cvs),
        batch_size(b_size), tanh_kern(tanh), hl_back_kern(hl_back) {
      Weights hidden_layer_weights(0, n_in * n_out, n_in + n_out);
      weights = new GpuWeights(hidden_layer_weights, cv, 0);
      Batch out_batch(b_size, 1, 1, n_out);
      output = new GpuBatch(out_batch, cv, 0);
      Batch v_batch(b_size, 1, 1, n_out);
      v = new GpuBatch(v_batch, cv, 0);
      Weights b(0, n_out, 0);
      bias = new GpuWeights(b, cv, 0);
      Batch delta(b_size, 1, 1, n_in);
      delta_bias = new GpuBatch(delta, cv, 0);
    }

    void forward(GpuBatch* batch) {
      cl_int err = CL_SUCCESS;
      for (int offset = 0; offset < batch_size; offset++) {
        err = clblasSgemv(clblasRowMajor, clblasNoTrans, num_out, num_in, 
                          1.0f, weights->buf, 0, num_in, batch->buf, 
                          offset * num_in, 1, 1.0f, v->buf, offset * num_out, 1,
                          1, &cv.commands[0], 0, NULL, NULL);
        CHK_ERR(err);
        err = clblasSaxpy(num_out, 1.0f, bias->buf, 0, 1, v->buf, 
            offset * num_out, 1, 1, &cv.commands[0], 0, NULL, NULL);
        CHK_ERR(err);
      }
      ocl_tanh(*output, *v, cv, tanh_kern, 0);
    }
    void backward(GpuBatch* prev_bias) {
      cl_int err = CL_SUCCESS;
      hl_back_ocl(*delta_bias, *v, hl_back_kern, cv);
      for (int offset = 0; offset < batch_size; offset++) {
        err = clblasSgemv(clblasRowMajor, clblasNoTrans, num_out, num_in, 
                          1.0f, weights->buf, 0, num_in, delta_bias->buf, 
                          offset * num_in, 1, 1.0f, prev_bias->buf, offset * num_in, 1,
                          1, &cv.commands[0], 0, NULL, NULL
            );
        CHK_ERR(err);
      }
    }
};

class OutputLayer {
  public:
    GpuWeights* weights;
    GpuWeights* bias;
    GpuBatch* output;
    GpuBatch* v;
    cl_vars_t cv;
    int batch_size;
    cl_kernel tanh_kern;
    int num_in;
    int num_out;
    GpuBatch* delta_bias;
    cl_kernel ol_back_kern;
    cl_kernel hl_back_kern;

    OutputLayer(int n_in, int n_out, int b_size, cl_vars_t cvs, cl_kernel tanh, 
        cl_kernel ol_back, cl_kernel hl_back): 
        num_in(n_in), num_out(n_out), cv(cvs), batch_size(b_size), tanh_kern(tanh),
        ol_back_kern(ol_back), hl_back_kern(hl_back) {
      Weights hidden_layer_weights(0, n_in * n_out, n_in + n_out);
      weights = new GpuWeights(hidden_layer_weights, cv, 0);
      Batch out_batch(b_size, 1, 1, n_out);
      output = new GpuBatch(out_batch, cv, 0);
      Batch v_batch(b_size, 1, 1, n_out);
      v = new GpuBatch(v_batch, cv, 0);
      Batch delta(b_size, 1, 1, n_in);
      delta_bias = new GpuBatch(delta, cv, 0);
    }

    void forward(GpuBatch* batch) {
      cl_int err = CL_SUCCESS;
      for (int offset = 0; offset < batch_size; offset++) {
        err = clblasSgemv(clblasRowMajor, clblasNoTrans, 
                          num_out, 
                          num_in, 
                          1.0f, weights->buf, 
                          0, num_in,
                          batch->buf, 
                          offset * num_in, 1,
                          1.0f, v->buf, 
                          offset * num_out, 1, 1,
                          &cv.commands[0], 0, NULL, NULL);
        CHK_ERR(err);
      }
      ocl_tanh(*output, *v, cv, tanh_kern, 0);
    }

    void backward(GpuBatch* actual, GpuBatch* prev_bias) {
      ocl_ol_back(*delta_bias, *actual, *output, *v, ol_back_kern, cv);
      cl_int err = CL_SUCCESS;
      for (int offset = 0; offset < batch_size; offset++) {
        err = clblasSgemv(clblasRowMajor, clblasNoTrans, num_out, num_in, 
                          1.0f, weights->buf, 0, num_in, delta_bias->buf, 
                          offset * num_in, 1, 1.0f, prev_bias->buf, offset * num_in, 1,
                          1, &cv.commands[0], 0, NULL, NULL
            );
        CHK_ERR(err);
      }
      hl_back_ocl(*delta_bias, *v, hl_back_kern, cv);
    }
};
