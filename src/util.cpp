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

static void drawHorizontalFlowHistogram(const Mat& flow, Mat& cflowmap, int step,
                                        double, const Scalar& color)
{
  for(int y = 0; y < cflowmap.rows; y += step) {
    float rowCount = 0;
    for(int x = 0; x < cflowmap.cols; x += step)
      {
        const Point2f& fxy = flow.at<Point2f>(y, x);
        rowCount += fxy.x + fxy.y;
      }
    line(cflowmap, Point(0,y), Point(cvRound(rowCount), y), color);
  }
  for(int x = 0; x < cflowmap.cols; x += step) {
    float colCount = 0;
    for(int y = 0; y < cflowmap.rows; y += step)
      {
        const Point2f& fxy = flow.at<Point2f>(y, x);
        colCount += fxy.x + fxy.y;
      }
    line(cflowmap, Point(x,0), Point(x, cvRound(colCount)), color);
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

