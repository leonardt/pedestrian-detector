__kernel void conv(__global float *output, __global const float *input, __constant float *weights, 
	           int rows, int cols, int depth, int num_sets, int n, __local float* buf) {
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
    float elt = FLT_MIN;
    #pragma unroll 2
    for (int ii = i * 2; ii < (i + 1) * 2; ii++) {
      __local float *buf_tmp = &buf[ii * (cols - 2 * 2)];
      #pragma unroll 2
      for (int jj = j * 2; jj < (j + 1) * 2; jj++) {
        elt = max(elt, buf_tmp[jj]);
      }
    }
    output_tmp[n * output_rows * output_cols + i * output_cols + j] = elt;
  }
}

void __kernel hidden_layer(__global float* output, __global float* v, __global const float* input, 
                           __constant float* weights, __constant float* bias,
                           int last_layer, int size, __local float* buf) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  float sum = 0.0f;
  for (int i = y; i < size; i+= get_global_size(1)) {
    sum += input[x * size + i] * weights[i];
  }
  buf[y] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (y == 0) {
    float out = bias[x];
    for (size_t i = 0; i < get_global_size(1); i++) {
      out += buf[i];
    }
    v[x] = out;
    if (last_layer) {
      output[x] = tanh(out);
    } else {
      output[x] = out;
    }
  }
}

void __kernel soft_max(__global float* in, __local float* buf) {
  size_t tid = get_local_id(0);
  size_t gid = get_group_id(0);
  size_t dim = get_local_size(0);
  size_t idx = get_global_id(0);
  buf[tid] = in[idx];
  barrier(CLK_LOCAL_MEM_FENCE);
  // Perform the reduction tree
  for (unsigned int s=dim/2; s > 0; s >>= 1) {
    // Reduce if thread is active for this level
    if (tid < s) {
      buf[tid] += exp(buf[tid + s]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  // Last thread writes the output
  in[idx] = exp(in[idx]) / buf[0];
}
