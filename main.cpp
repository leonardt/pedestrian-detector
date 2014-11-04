#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <cstdlib>

using namespace cv;
using namespace std;

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
                    double, const Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}

static void drawHSVFlow(const Mat& flow, Mat& cflowmap) {
  cv::Mat xy[2];
  cv::split(flow, xy);
  cv::Mat magnitude, angle;
  cv::cartToPolar(xy[0], xy[1], magnitude, angle, true);

  //translate magnitude to range [0;1]
  double mag_max;
  cv::minMaxLoc(magnitude, 0, &mag_max);
  magnitude.convertTo(magnitude, -1, 1.0/mag_max);

  //build hsv image
  cv::Mat _hsv[3], hsv;
  _hsv[0] = angle;
  _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
  _hsv[2] = magnitude;
  cv::merge(_hsv, 3, hsv);

  //convert to BGR and show
  cv::cvtColor(hsv, cflowmap, cv::COLOR_HSV2BGR);
}

int openWebcam(VideoCapture cap) {
  cap.open(0);
  if( !cap.isOpened() ) {
    cout << "Couldn't open webcam, exiting.\n";
    return 1;
  }
  return 0;
}


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

  Mat prevgray, gray, flow, cflow, frame;
  namedWindow("flow", 1);

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
      if (count < 0) {
        count++;
      } else {
        count = 0;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        if( prevgray.data ) {
            /* calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 5, 5, 1.1, 0); */
            tvl1->calc(prevgray, gray, flow);
            if (mode == 1) {
              drawOptFlowMap(flow, cflow, 10, 1.5, Scalar(0, 255, 0));
            } else {
              drawHSVFlow(flow, cflow);
              imshow("orig", gray);
            }
            imshow("flow", cflow);
        }
        if(waitKey(30)>=0)
            break;
        std::swap(prevgray, gray);
      }
  }
  return 0;
}
