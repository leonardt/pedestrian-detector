#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <cstdlib>

#include "src/util.cpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
  VideoCapture cap;
  if (argc > 1) {
    cap.open(argv[1]);
    if( !cap.isOpened() ) {
      cout << "Couldn't open video %s, falling back to webcam\n" << argv[1];
      int err = openWebcam(cap);
      if (err) { return -1; }
    }
  } else{
    cout << "No video provided, using webcam.\n";
    int err = openWebcam(cap);
    if (err) { return -1; }
  }

  Mat prevgray, gray, flow, cflow, frame, hist;
  namedWindow("flow", 1);
  namedWindow("hist", 1);

  int mode = 0;

  if (argc > 2) {
    mode = atoi(argv[2]);
  } else {
    namedWindow("orig", 1);
    moveWindow("orig", 0, 440);
  }
  Ptr<DenseOpticalFlow> tvl1 = createOptFlow_DualTVL1();

  int count = 0;
  while(true) {
      cap >> frame;
      if (frame.empty()) {
        break;
      }
      count++;
      if (count == 2) break;
      count = 0;
      cvtColor(frame, gray, COLOR_BGR2GRAY);

      if( prevgray.data ) {
        calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 5, 5, 1.1, 0);
        // tvl1->calc(prevgray, gray, flow);
        cvtColor(frame, hist, COLOR_BGR2GRAY);
        if (mode == 1) {
          cvtColor(frame, cflow, COLOR_BGR2GRAY);
          drawOptFlowMap(flow, cflow, 10, 1.5, Scalar(0, 255, 0));
        } else {
          drawHSVFlow(flow, cflow);
          imshow("orig", gray);
        }
        imshow("flow", cflow);
        // drawHorizontalFlowHistogram(flow, hist, 10, 1.5, Scalar(0, 255, 0));
        // imshow("horiz", hist);
      }
      if(waitKey(30)>=0)
          break;
      std::swap(prevgray, gray);
  }
  return 0;
}
