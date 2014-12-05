__kernel void conv(__global float *output, __global float *Input, __constant float *Weights, 
	 int rows, int cols, int depth, int r, int s, int num_sets, __local float* buf, __local float* buf2) {
  int i = get_global_id(0);
  int j = get_global_id(1);
  int z = get_global_id(2);
  for (int d = 0; d < depth; d++) {
    buf[d * rows * cols + i * cols + j] = Input[z * rows * cols * depth + d * rows * cols + i * cols + j];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int n = 0; n < num_sets; n++) {
    if ((i >= r && i < rows - r) && (j >= r && j < cols - r)) {
      float elt;
      float weight;

      float sum = 0.0f;
      for (int d = 0; d < depth; d++) {
        for (int ii = -r; ii <= r; ii++) {
          for (int jj = -r; jj <= r; jj++) {
            elt = buf[d * rows * cols + (i + ii) * cols + j + jj];
            weight = Weights[n * depth * (2 * r + 1) * (2 * r + 1) + 
              d * (2 * r + 1) * (2 * r + 1) +
              (ii + r) * (2 * r + 1) + jj + r];
            sum += weight * elt;
          }
        }
      }
      buf2[(i - r) * (cols - 2 * r) + (j - r)] = sum;
    }
    int output_rows = (rows - 2 * r) / s;
    int output_cols = (cols - 2 * r) / s;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (i < output_rows && j < output_cols) {
      float elt = FLT_MIN;
      for (int ii = i * s; ii < (i + 1) * s; ii++) {
        for (int jj = j * s; jj < (j + 1) * s; jj++) {
          float curr = buf2[ii * (cols - 2 * r) + jj];
          if (curr > elt) {
            elt = curr;
          }
        }
      }
      output[z * num_sets * output_rows * output_cols + 
        n * output_rows * output_cols + 
        i * output_cols + j] = elt;
    }
  }
}

__kernel void max_pool(__global float *Output, __global float *Input, int rows, int cols, int s) {

  int i = get_global_id(0);
  int j = get_global_id(1);
  int z = get_global_id(2);

  float elt = FLT_MIN;
  
  for (int ii = i * s; ii < (i + 1) * s; ii++) {
    for (int jj = j * s; jj < (j + 1) * s; jj++) {
      float curr = Input[z * rows * cols + ii * cols + jj];
      if (curr > elt) {
        elt = curr;
      }
    }
  }
  Output[z * get_global_size(0) * get_global_size(1) + i * get_global_size(1) + j] = elt;
}
