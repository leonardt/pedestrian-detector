#include <iostream>
#include <assert.h>
#include <limits>
#include <vector>

#include "opencv2/video/tracking.hpp"

using namespace cv;

void convolve(float* input, float* output, float* weights, int n, int r) {
  for (int i = r; i < n - r; i++) {
    for (int j = r; j < n - r; j++) {


      float sum = 0.0;
      for (int ii = -r; ii <= r; ii++) {
        for (int jj = -r; jj <= r; jj++) {
          float elt = input[(i + ii) * n + j + jj];
          //printf("elt: %f\n", elt);
          float weight = weights[(ii + r) * (2 * r + 1) + jj + r];
          sum += weight * elt;
        }
      }
      output[(i - r) * (n - 2 * r) + (j - r)] += sum;
      //printf("%f\n", sum);
    }
  }
}


// if going from conv to max pool, will want to keep result of conv on gpu to avoid memory overhead

void max_pool(float* input, float* output, int n, int s) {
  for (int i = 0; i < n / s; i++) {
    for (int j = 0; j < n / s; j++) {
      float elt = std::numeric_limits<float>::min();
      for (int ii = i * s; ii < i * s + s; ii++) {
        for (int jj = j * s; jj < j * s + s; jj++) {
          float curr = input[ii * n + jj];
          printf("%0.3f\n", curr);
          if (curr > elt) {
            elt = curr;
          }
        }
      }
      output[i * n + j] = elt;
    }
  }
}

/*
void convolution_layer(vector<Mat> inputs, 
                       vector<Mat> outputs,
                       vector<Mat> weights) {
  assert(outputs.size() == weights.size());
  for (uint8_t i = 0; i < weights.size(); i++) {
    for (uint8_t j = 0; j < inputs.size(); j++) {
      convolve(inputs[i], weights[i], )
    }
  }

}

int main(int argc, char *argv[]) {
  int n = 6;
  float* input = new float[n * n];
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      input[i * n + j] = i * n + j;
    }
  }
  printf("Input\n");
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%0.3f ", input[i * n + j]);
    }
    printf("\n");
  }
  printf("------------\n");
  int w = 2;
  float* weights = new float[(2 * w + 1) * (2 * w + 1)];
  for (int i = 0; i < 2 * w + 1; ++i) {
    for (int j = 0; j < 2 * w + 1; ++j) {
      weights[i * (2 * w + 1) + j] = .25;
    }
  }
  int out_n = n - 2 * w;
  float* output = new float[out_n * out_n];
  for (int i = 0; i < out_n; ++i) {
    for (int j = 0; j < out_n; ++j) {
      output[i * out_n + j] = 0.0;
    }
  }
  convolve(input, output, weights, n, 2);
  printf("Output\n");
  for (int i = 0; i < out_n; ++i) {
    for (int j = 0; j < out_n; ++j) {
      printf("%0.3f ", output[i * out_n + j]);
    }
    printf("\n");
  }
  printf("------------\n");
  int pool_n = out_n / 2;
  float* pooled = new float[pool_n * pool_n];
  max_pool(output, pooled, out_n, 2);
  printf("Pooled\n");
  for (int i = 0; i < pool_n; ++i) {
    for (int j = 0; j < pool_n; ++j) {
      printf("%0.3f ", pooled[i * pool_n + j]);
    }
    printf("\n");
  }
  printf("------------\n");
  return 0;
}
*/