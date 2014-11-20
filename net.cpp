#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <random>

using namespace cv;
using namespace std;


void convolve(float* input, float* output, int rows, int cols, float* weights, int r) {
  for (int i = r; i < rows - r; i++) {
    for (int j = r; j < cols - r; j++) {
      float sum = 0.0;
      for (int ii = -r; ii <= r; ii++) {
        for (int jj = -r; jj <= r; jj++) {
          float elt = input[(i + ii) * cols + j + jj];
          float weight = weights[(ii + r) * (2 * r + 1) + jj + r];
          sum += weight * elt;
        }
      }
      output[(i - r) * (cols - 2 * r) + (j - r)] = sum;
    }
  }
}

void layer1_compute(float* input, float* output, int rows, int cols, float* weights, int r) {
  convolve(input, output, rows, cols, weights, r);
}

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    float* img_data = (float*)image.data;

    int l1_numoutputs = 4;
    vector<float*> l1_outputs(0);
    int w = 2;
    float* l1_weights = new float[(w * 2 + 1) * (w * 2 + 1)];
    for (int i = 0; i < (w * 2 + 1); i++) {
       for (int j = 0; j < (w * 2 + 1); j++) {
         l1_weights[i * (w * 2 + 1) + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      }
    }
    for (int i = 0; i < l1_numoutputs; i++) {
      float* out = new float[image.rows * image.cols];
      layer1_compute(img_data, out, image.rows, image.cols, l1_weights, w);
      l1_outputs.push_back(out);
    }

    // namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    // imshow( "Display window", image );                   // Show our image inside it.

    // waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
