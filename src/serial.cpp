using namespace cv;


void convolve(Batch input, Batch output, float *weights, int r) {
  for (int z = 0; z < input.batch_size; z++) {
    for (int i = r; i < input.rows - r; i++) {
        for (int j = r; j < input.cols - r; j++) {
            float sum = 0.0;
            for (int ii = -r; ii <= r; ii++) {
                for (int jj = -r; jj <= r; jj++) {
                    float elt = input.data[z * input.rows * input.cols +
                                          (i + ii) * input.cols + j + jj];
                    float weight = weights[(ii + r) * (2 * r + 1) + jj + r];
                    sum += weight * elt;
                }
            }
            output.data[z * output.rows * output.cols + (i - r) * output.cols + (j - r)] += sum;
        }
    }
  }
}

void max_pool(Batch input, Batch output, int s) {
  for (int z = 0; z < input.batch_size; z++) {
    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            float elt = std::numeric_limits<float>::min();
            for (int ii = i * s; ii < (i + 1) * s; ii++) {
                for (int jj = j * s; jj < (j + 1) * s; jj++) {
                    float curr = input.data[z * input.rows * input.cols + ii * input.cols + jj];
                    if (curr > elt) {
                        elt = curr;
                    }
                }
            }
            output.data[z * output.rows * output.cols + i * output.cols + j] = elt;
        }
    }
  }
}

void layer1_compute(Batch input, Batch output, float *weights, int r, int s) {
    Batch conv_out(input.batch_size, input.rows - 2 * r, input.cols - 2 * r);
    convolve(input, conv_out, weights, r);
    max_pool(conv_out, output, s);
}

void layer2_compute(vector<Batch> inputs, Batch output, vector<float *> weights, int r, int s) {
    int rows = inputs[0].rows;
    int cols = inputs[0].cols;
    Batch conv_out(inputs[0].batch_size, rows - 2 * r, cols - 2 * r);
    for (int i = 0; i < weights.size(); ++i) {
        convolve(inputs[i], conv_out, weights[i], r);
    }
    max_pool(conv_out, output, s);
}
