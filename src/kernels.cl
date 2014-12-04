__kernel void conv(__global float *Output, __global float *Input, __global float *Weights, 
	 int rows, int cols, int r, __local float* buf) {

  // weights is passed in, read only, read by all work items
  // optimization: load weights into shared memory
  // can use constant memory since it's read only


  // will want to prefetch into local memory, basically avoid going to global memory



  // use this serial version as correctness test
  // start with a naive opencl impl, (use the hw clhelper code)
  // then slowly add optimizations and make sure it stays correct

  // for input, expect a pointer to matrix of floats, and something about size/dimensions

  // Retrieve global index values for each dimension. These are used to index into the matrices
  int i = get_global_id(0);
  int j = get_global_id(1);
  int z = get_global_id(2);
  buf[i * cols + j] = Input[z * rows * cols + i * cols + j];
  barrier(CLK_LOCAL_MEM_FENCE);

  if ((i >= r && i < rows - r) && (j >= r && j < cols - r)) {

    float sum = 0.0f;
    float elt;
    float weight;

    for (int ii = -r; ii <= r; ii++) {
      for (int jj = -r; jj <= r; jj++) {
        elt = buf[(i + ii) * cols + j + jj];
        weight = Weights[(ii + r) * (2 * r + 1) + jj + r];
        sum += weight * elt;
      }
    }
    Output[z * (rows - 2 * r) * (cols - 2 * r) + (i - r) * (cols - 2 * r) + (j - r)] += sum;
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
