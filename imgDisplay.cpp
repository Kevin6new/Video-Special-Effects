/* 
Basil Reji & Kevin Sebastian Sani
Spring 2024
Pattern Recognition & Computer Vision
Assignment 1 - Video Special Effects
Image Display
*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;

int imgDisplay()
{
    cv::Mat img = cv::imread("D:/Opencv/Video_Special_Effects/basil.jpg");
    namedWindow("First OpenCV Application", WINDOW_AUTOSIZE);
    cv::imshow("First OpenCV Application", img);
    cv::moveWindow("First OpenCV Application", 0, 45);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
