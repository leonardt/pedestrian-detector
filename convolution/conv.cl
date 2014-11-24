__kernel void conv(__global float *Output, __global float *Input, __global float *Weights, 
	 int n, int r)
{

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

  if ((i >= r && i < n - r) && (j >= r && j < n - r)) {

    float sum = 0.0f;
    float elt;
    float weight;

    for (int ii = -r; ii <= r; ii++) {
      for (int jj = -r; jj <= r; jj++) {
        elt = Input[(i + ii) * n + j + jj];
        weight = Weights[(ii + r) * (2 * r + 1) + jj + r];
        sum += weight * elt;
      }
    }
    Output[(i - r) * (n - 2 * r) + (j - r)] += sum;
  }
}
