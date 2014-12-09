#include <omp.h>
#include <math.h>
#include "types.h"

using namespace std;


void convolve(Batch input, Batch output, vector<Weights> weights_sets, int r, vector<float> bias) {
  for (int z = 0; z < input.batch_size; z++) {
    for (int i = r; i < input.rows - r; i++) {
      for (int j = r; j < input.cols - r; j++) {
        for (int n = 0; n < weights_sets.size(); n++) {
          float sum = 0.0;
          Weights weights = weights_sets[n];
          for (int k = 0; k < input.depth; k++) {
            for (int ii = -r; ii <= r; ii++) {
              for (int jj = -r; jj <= r; jj++) {
                float elt = input.data[z * input.depth * input.rows * input.cols +
                                       k * input.rows * input.cols +
                                      (i + ii) * input.cols + j + jj];
                float weight = weights.data[k * weights.rows * weights.cols +
                                            (ii + r) * weights.cols + jj + r];
                sum += weight * elt;
              }
            }
          }
          output.data[z * output.depth * output.rows * output.cols +
                      n * output.rows * output.cols +
                      (i - r) * output.cols + (j - r)] = tanh(sum + bias[n]);
        }
      }
    }
  }
}

void max_pool(Batch input, Batch output, int s) {
  for (int z = 0; z < input.batch_size; z++) {
    for (int i = 0; i < output.rows; i++) {
      for (int j = 0; j < output.cols; j++) {
        for (int d = 0; d < input.depth; d++) {
          float elt = input.data[z * input.depth * input.rows * input.cols + 
                d * input.rows * input.cols + i * s * input.cols + j * s];
          for (int ii = i * s; ii < (i + 1) * s; ii++) {
            for (int jj = j * s; jj < (j + 1) * s; jj++) {
              float curr = input.data[z * input.depth * input.rows * input.cols + 
                d * input.rows * input.cols + ii * input.cols + jj];
              if (curr > elt) {
                elt = curr;
              }
            }
          }
          output.data[z * output.depth * output.rows * output.cols + 
                      d * output.rows * output.cols + 
                      i * output.cols + j] = elt;
        }
      }
    }
  }
}

void layer1_compute(Batch input, Batch output, std::vector<Weights> weights, 
                    int r, int s, vector<float> bias) {
    Batch conv_out(input.batch_size, weights.size(), input.rows - 2 * r, input.cols - 2 * r);
    convolve(input, conv_out, weights, r, bias);
    max_pool(conv_out, output, s);
}
