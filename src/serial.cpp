using namespace cv;


void convolve(Mat input, Mat output, float *weights, int r) {
    for (int i = r; i < input.rows - r; i++) {
        for (int j = r; j < input.cols - r; j++) {
            float sum = 0.0;
            for (int ii = -r; ii <= r; ii++) {
                for (int jj = -r; jj <= r; jj++) {
                    float elt = input.at<float>(i + ii, j + jj);
                    float weight = weights[(ii + r) * (2 * r + 1) + jj + r];
                    sum += weight * elt;
                }
            }
            output.at<float>(i - r, j - r) += sum;
        }
    }
}

void max_pool(Mat input, Mat output, int s) {
    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            float elt = std::numeric_limits<float>::min();
            for (int ii = i * s; ii < (i + 1) * s; ii++) {
                for (int jj = j * s; jj < (j + 1) * s; jj++) {
                    float curr = input.at<float>(ii, jj);
                    if (curr > elt) {
                        elt = curr;
                    }
                }
            }
            output.at<float>(i, j) = elt;
        }
    }
}

void layer1_compute(Mat input, Mat output, float *weights, int r, int s) {
    Mat conv_out = Mat::zeros(input.rows - 2 * r, input.cols - 2 * r, CV_32F);
    convolve(input, conv_out, weights, r);
    max_pool(conv_out, output, s);
}

Mat layer2_compute(vector<Mat> inputs, vector<float *> weights, int r, int s) {
    int rows = inputs[0].rows;
    int cols = inputs[0].cols;
    Mat conv_out = Mat::zeros(rows - 2 * r, cols - 2 * r, CV_32F);
    for (int i = 0; i < weights.size(); ++i) {
        convolve(inputs[i], conv_out, weights[i], r);
    }
    Mat output = Mat::zeros((rows - 2 * r) / s, (cols - 2 * r) / s, CV_32F);
    max_pool(conv_out, output, s);
    return output;
}
