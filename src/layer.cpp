#include <cstdlib>
#include "conv.cpp"
#include "serial.cpp"
#include "types.h"

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
    vector<vector<GpuWeights> > weights;
    vector<float> bias;
    int num_outputs;
    int num_inputs;
    int w;
    cl_kernel conv;
    GpuBatch* output;
    cl_vars_t cv;

    GpuConvPoolLayer(ConvPoolLayer layer, cl_vars_t cvs, cl_kernel kernel): cv(cvs), conv(kernel) {
      num_outputs = layer.num_outputs;
      num_inputs = layer.num_inputs;
      w = layer.w;
      for (cl_uint device = 0; device < cv.num_devices; device++) {
        vector<GpuWeights> weights_set;
        for (Weights w : layer.weights) {
          weights_set.push_back(GpuWeights(w, cv, device));
          bias.push_back(0.0f);
        }
        weights.push_back(weights_set);
        output = new GpuBatch(*layer.output, cv, device);
      }
    }

    void forward(GpuBatch* batch, cl_uint device) {
      ocl_conv(*batch, *output, weights[device], bias, w, cv, conv, device);
      cl_int err = CL_SUCCESS;
      err = clFinish(cv.commands[device]);
    }
};
