#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <random>
#include "src/conv/conv.cpp"
#include <cstdlib>
#include <cstdlib>

using namespace cv;
using namespace std;


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
    /* cout << "convolved = " << endl << " " << conv_out << endl << endl; */
    max_pool(conv_out, output, s);
}

void layer1_ocl(Mat input, Mat output, float *weights, int r, int s, cl_vars_t cv, cl_kernel conv) {
    Mat conv_out = Mat::zeros(input.rows - 2 * r, input.cols - 2 * r, CV_32F);
    ocl_conv(input, conv_out, weights, r, cv, conv);
    /* cout << "convolved = " << endl << " " << conv_out << endl << endl; */
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

Mat layer2_ocl(vector<Mat> inputs, vector<float *> weights, int r, int s, cl_vars_t cv, cl_kernel conv) {
    int rows = inputs[0].rows;
    int cols = inputs[0].cols;
    Mat conv_out = Mat::zeros(rows - 2 * r, cols - 2 * r, CV_32F);
    for (int i = 0; i < weights.size(); ++i) {
      ocl_conv(inputs[i], conv_out, weights[i], r, cv, conv);
    }
    Mat output = Mat::zeros((rows - 2 * r) / s, (cols - 2 * r) / s, CV_32F);
    max_pool(conv_out, output, s);
    return output;
}

float *gen_random_weights(int radius, float range) {
    float *weights = new float[(radius * 2 + 1) * (radius * 2 + 1)];
    for (int i = 0; i < (radius * 2 + 1); i++) {
        for (int j = 0; j < (radius * 2 + 1); j++) {
            weights[i * (radius * 2 + 1) + j] = ((float) rand() / (float) RAND_MAX) * (range * 2 + 1) - range;
        }
    }
    return weights;
}

void check_outputs(Mat ocl_out, Mat out) {
  for (int i = 0; i < ocl_out.rows; i++) {
    for (int j = 0; j < ocl_out.cols; j++) {
      if (ocl_out.at<float>(i, j) != out.at<float>(i, j)) {
        cout << "Not equal at (" << i << ", " << j << ")" << endl;
        cout << "Serial" << endl;
        cout << out << endl;
        cout << "OCL" << endl;
        cout << ocl_out;
        exit(1);
      }
    }
  }
}

int main(int argc, char **argv) {
    srand(time(NULL));
    if (argc != 2) {
        cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }

    Mat image;
    imread(argv[1], 0).convertTo(image, CV_32F);

    if (!image.data) {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // cout << "img_data = " << endl << " " << image << endl << endl;

    std::string conv_kernel_str;

    std::string conv_name_str =
            std::string("conv");
    std::string conv_kernel_file =
            std::string("./src/conv/conv.cl");

    cl_vars_t cv;
    cl_kernel conv;

    readFile(conv_kernel_file,
            conv_kernel_str);

    initialize_ocl(cv);

    compile_ocl_program(conv, cv, conv_kernel_str.c_str(),
            conv_name_str.c_str());

    int l1_numoutputs = 4;
    vector<Mat> l1_outputs(0);
    int w = 2;
    float *l1_weights = gen_random_weights(w, 1);

    // convolution layer and pooling
    for (int i = 0; i < l1_numoutputs; i++) {
        Mat out = Mat::zeros((image.rows - 2 * w) / 2, (image.cols - 2 * w) / 2, CV_32F);
        Mat ocl_out = Mat::zeros((image.rows - 2 * w) / 2, (image.cols - 2 * w) / 2, CV_32F);
        double t0 = timestamp();
        layer1_compute(image, out, l1_weights, w, 2);
        t0 = timestamp() - t0;
        cout << "layer1 naive time " << t0 << endl;
        
        t0 = timestamp();
        layer1_ocl(image, ocl_out, l1_weights, w, 2, cv, conv);
        t0 = timestamp() - t0;
        cout << "layer1 ocl time " << t0 << endl;
        check_outputs(ocl_out, out);
        
        l1_outputs.push_back(ocl_out);
    }

    vector<vector<float *> > l2_weights(0);
    int l2_numoutputs = 6;
    for (int i = 0; i < l2_numoutputs; ++i) {
        vector<float *> weights(0);
        for (int j = 0; j < l1_numoutputs; ++j) {
            weights.push_back(gen_random_weights(2, l1_numoutputs));
        }
        l2_weights.push_back(weights);
    }

    vector<Mat> l2_outputs(0);
    for (vector<float *> weights : l2_weights) {
        double t0 = timestamp();
        Mat out = layer2_compute(l1_outputs, weights, 2, 2);
        t0 = timestamp() - t0;
        cout << "layer2 naive time " << t0 << endl;
        t0 = timestamp();
        Mat ocl_out = layer2_ocl(l1_outputs, weights, 2, 2, cv, conv);
        t0 = timestamp() - t0;
        cout << "layer2 ocl time " << t0 << endl;
        l2_outputs.push_back(ocl_out);
        check_outputs(ocl_out, out);
    }

    // namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    // imshow( "Display window", image );                   // Show our image inside it.

    // waitKey(0);                                          // Wait for a keystroke in the window
    cout << "PASSED" << endl;
    uninitialize_ocl(cv);
    
    return 0;
}