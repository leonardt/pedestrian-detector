__kernel void conv(__global float *output, __global const float *input, __constant float *weights, 
	           float bias, int rows, int cols, int depth, int num_sets, int n, __local float* buf) {
  int i = get_global_id(0);
  int j = get_global_id(1);
  int z = get_global_id(2);
  int depth_stride = rows * cols;
  int weight_stride = 2 * 2 + 1;
  int output_rows = (rows - 2 * 2) / 2;
  int output_cols = (cols - 2 * 2) / 2;
  __global float* output_tmp = &output[z * num_sets * output_rows * output_cols];
  float sum = 0.0f;
  for (int d = 0; d < depth; d++) {
    buf[i * cols + j] = input[z * depth_stride * depth +
                              d * depth_stride + i * cols + j];
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((i >= 2 && i < rows - 2) && (j >= 2 && j < cols - 2)) {
      __constant float* weights_tmp = &weights[d * weight_stride * weight_stride];
      #pragma unroll 
      for (int ii = -2; ii <= 2; ii++) {
        __local float* buf1_tmp = &buf[(i + ii) * cols];
        __constant float* weights_tmp1 = &weights_tmp[(ii + 2) * weight_stride];
        #pragma unroll 
        for (int jj = -2; jj <= 2; jj++) {
          float elt = buf1_tmp[j + jj];
          float weight = weights_tmp1[jj + 2];
          sum += weight * elt;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((i >= 2 && i < rows - 2) && (j >= 2 && j < cols - 2)) {
    buf[(i - 2) * (cols - 2 * 2) + (j - 2)] = sum;
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  if (i < output_rows && j < output_cols) {
    float elt = buf[i * 2 * (cols - 2 * 2) + j * 2];
    #pragma unroll 2
    for (int ii = i * 2; ii < (i + 1) * 2; ii++) {
      __local float *buf_tmp = &buf[ii * (cols - 2 * 2)];
      #pragma unroll 2
      for (int jj = j * 2; jj < (j + 1) * 2; jj++) {
        elt = max(elt, buf_tmp[jj]);
      }
    }
    output_tmp[n * output_rows * output_cols + i * output_cols + j] = tanh(elt + bias);
  }
}

void __kernel ocl_tan_h(__global float* a, __global float* v) {
  int i = get_global_id(0);
  a[i] = tanh(v[i]);
}

inline float tanh_prime(float input) {
    return (4*pow(cosh(input), 2)) / pow((cosh(2*input)+1), 2);
}

void __kernel ol_back(__global float* delta, __global float* actual, __global float* prev_out,
                               __global float* prev_v) {
  int i = get_global_id(0);
  delta[i] = (prev_out[i] - actual[i]) * tanh_prime(prev_v[i]);
}

void __kernel hl_back(__global float* delta, __global float* v) {
  int i = get_global_id(0);
  delta[i] *= tanh_prime(v[i]);
}
