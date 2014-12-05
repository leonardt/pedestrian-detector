__kernel void conv(__global float *output, __global const float *input, __constant float *weights, 
	                 int rows, int cols, int depth, int r, int s, int num_sets,
                   __local float* buf, __local float* buf2) {
  int i = get_global_id(0);
  int j = get_global_id(1);
  int z = get_global_id(2);
  int depth_stride = rows * cols;
  int weight_stride = 2 * r + 1;
  int output_rows = (rows - 2 * r) / s;
  int output_cols = (cols - 2 * r) / s;
  for (int d = 0; d < depth; d++) {
    buf[d * depth_stride + i * cols + j] = input[z * depth_stride * depth +
                                                d * depth_stride + i * cols + j];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  __global float* output_tmp = &output[z * num_sets * output_rows * output_cols];
  for (int n = 0; n < num_sets; n++) {
    __constant float* weights_set = &weights[n * depth * weight_stride * weight_stride];
    if ((i >= r && i < rows - r) && (j >= r && j < cols - r)) {

      float sum = 0.0f;
      for (int d = 0; d < depth; d++) {
        __local float* buf1_tmp = &buf[d * depth_stride];
        __constant float* weights_tmp = &weights_set[d * weight_stride * weight_stride];
        for (int ii = -r; ii <= r; ii++) {
          __local float* buf1_tmp1 = &buf1_tmp[(i + ii) * cols];
          __constant float* weights_tmp1 = &weights_tmp[(ii + r) * weight_stride];
          for (int jj = -r; jj <= r; jj++) {
            float elt = buf1_tmp1[j + jj];
            float weight = weights_tmp1[jj + r];
            sum += weight * elt;
          }
        }
      }
      buf2[(i - r) * (cols - 2 * r) + (j - r)] = sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (i < output_rows && j < output_cols) {
      float elt = FLT_MIN;
      for (int ii = i * s; ii < (i + 1) * s; ii++) {
        __local float *buf2_tmp = &buf2[ii * (cols - 2 * r)];
        for (int jj = j * s; jj < (j + 1) * s; jj++) {
          elt = max(elt, buf2_tmp[jj]);
        }
      }
      output_tmp[n * output_rows * output_cols + i * output_cols + j] = elt;
    }
  }
}
