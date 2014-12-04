#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <random>
#include "src/conv.cpp"
#include "src/serial.cpp"
// #include "src/types.cpp"
#include <cstdlib>
#include <cstdlib>

using namespace cv;
using namespace std;

void layer1_ocl(GpuMat input, cl_mem output, cl_mem weights, int r, int s, cl_vars_t cv, cl_kernel conv, cl_kernel pool) {
    cl_int err = CL_SUCCESS;
    cl_mem conv_out = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
                                     (input.rows - 2 * r) * (input.cols - 2 * r)
                                     * sizeof(float), NULL, &err);
    CHK_ERR(err);
    ocl_conv(input, conv_out, weights, r, cv, conv);
    cl_max_pool(conv_out, input.cols - 2 * r, (input.rows - 2 * r) / 2, (input.cols - 2 * r) / 2, output, s, cv, pool);

}

Mat layer2_ocl(vector<GpuMat> inputs, vector<cl_mem> weights, int r, int s, cl_vars_t cv, cl_kernel conv) {
  int rows = inputs[0].rows;
  int cols = inputs[0].cols;
    Mat conv_out = Mat::zeros(rows - 2 * r, cols - 2 * r, CV_32F);
    cl_int err = CL_SUCCESS;
    cl_mem g_Output = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
            conv_out.rows * conv_out.cols * sizeof(float), NULL, &err);
    CHK_ERR(err);
    for (size_t i = 0; i < weights.size(); ++i) {
        ocl_conv(inputs[i], g_Output, weights[i], r, cv, conv);
    }
    err = clEnqueueReadBuffer(cv.commands, g_Output, true, 0, conv_out.rows * conv_out.cols * sizeof(float),
            (float*)conv_out.data, 0, NULL, NULL);
    CHK_ERR(err);
    err = clFinish(cv.commands);
    CHK_ERR(err);
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

    cl_vars_t cv;
    initialize_ocl(cv);

    cl_kernel conv = build_kernel(string("conv"), string("./src/kernels.cl"), cv);
    cl_kernel pool = build_kernel(string("max_pool"), string("./src/kernels.cl"), cv);

    cl_int err = CL_SUCCESS;

    GpuMat ocl_image = GpuMat(image, cv);

    int l1_numoutputs = 4;
    vector<Mat> l1_outputs;
    vector<GpuMat> l1_ocl_outputs;
    int w = 2;
    float *l1_weights = gen_random_weights(w, 1);
    cl_mem ocl_l1_weights = clCreateBuffer(cv.context, CL_MEM_READ_ONLY,
                                           (w * 2 + 1) * (w * 2 + 1) *
                                           sizeof(float), NULL, &err);
    CHK_ERR(err);
    err = clEnqueueWriteBuffer(cv.commands, ocl_l1_weights, true, 0,
                               (w * 2 + 1) * (w * 2 + 1) *
                               sizeof(float), l1_weights, 0, NULL,
                               NULL);
    CHK_ERR(err);

    // convolution layer and pooling
    for (int i = 0; i < l1_numoutputs; i++) {
        Mat ocl_out = Mat::zeros((image.rows - 2 * w) / 2, (image.cols - 2 * w) / 2, CV_32F);
        Mat out = Mat::zeros((image.rows - 2 * w) / 2, (image.cols - 2 * w) / 2, CV_32F);
        double t0 = timestamp();
        layer1_compute(image, out, l1_weights, w, 2);
        t0 = timestamp() - t0;
        cout << "layer1 naive time " << t0 << endl;
        
        cl_mem ocl_out_buf = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
                out.rows * out.cols * sizeof(float), NULL, &err);
        CHK_ERR(err);
        t0 = timestamp();
        layer1_ocl(ocl_image, ocl_out_buf,
                   ocl_l1_weights, w, 2, cv, conv, pool);
        t0 = timestamp() - t0;
        cout << "layer1 ocl time " << t0 << endl;
        err = clEnqueueReadBuffer(cv.commands, ocl_out_buf, true, 0,
                                  out.rows * out.cols * sizeof(float),
                                  (float*)ocl_out.data, 0, NULL,
                                  NULL);
        CHK_ERR(err);
        check_outputs(ocl_out, out);
        
        l1_outputs.push_back(out);
        l1_ocl_outputs.push_back(GpuMat(ocl_out_buf, out.rows, out.cols));
    }

    vector<vector<float *> > l2_weights(0);
    vector<vector<cl_mem> > ocl_l2_weights(0);
    int l2_numoutputs = 6;
    for (int i = 0; i < l2_numoutputs; ++i) {
        vector<float *> weights;
        vector<cl_mem> ocl_weights;
        for (int j = 0; j < l1_numoutputs; ++j) {
            weights.push_back(gen_random_weights(2, l1_numoutputs));
            cl_mem ocl_weight = clCreateBuffer(cv.context, CL_MEM_READ_ONLY,
                                               (w * 2 + 1) * (w * 2 + 1) * sizeof(float), NULL, &err);
            CHK_ERR(err);
            err = clEnqueueWriteBuffer(cv.commands, ocl_weight, true, 0, 
                                       (w * 2 + 1) * (w * 2 + 1) * sizeof(float), 
                                       weights.back(), 0, NULL, NULL);
            CHK_ERR(err);
            ocl_weights.push_back(ocl_weight);
        }
        l2_weights.push_back(weights);
        ocl_l2_weights.push_back(ocl_weights);
    }

    vector<Mat> l2_outputs;
    vector<Mat> ocl_l2_outputs;
    for (vector<float *> weights : l2_weights) {
        double t0 = timestamp();
        Mat out = layer2_compute(l1_outputs, weights, 2, 2);
        t0 = timestamp() - t0;
        cout << "layer2 naive time " << t0 << endl;
        l2_outputs.push_back(out);
    }
    for (vector<cl_mem> weights : ocl_l2_weights) {
        double t0 = timestamp();
        Mat ocl_out = layer2_ocl(l1_ocl_outputs, weights, 2,
                                 2, cv, conv);
        t0 = timestamp() - t0;
        cout << "layer2 ocl time " << t0 << endl;
        ocl_l2_outputs.push_back(ocl_out);
    }
    for (size_t i = 0; i < l2_outputs.size(); i++) {
      check_outputs(ocl_l2_outputs[i], l2_outputs[i]);
    }

    cout << "PASSED" << endl;
    
    return 0;
}
