/* 
Basil Reji & Kevin Sebastian Sani
Spring 2024
Pattern Recognition & Computer Vision
Assignment 1 - Video Special Effects
Video Display file
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include "filter.h"
#pragma once
#include <ctime>
#include <opencv2/highgui/highgui.hpp>
#define FACE_CASCADE_FILE "D:/Opencv/Video_Special_Effects/haarcascade_frontalface_alt2.xml"

// Initialising variables 
int greyscale(cv::Mat& src, cv::Mat& dst);
int sepiaTone(cv::Mat& src, cv::Mat& dst);
int blur5x5(cv::Mat& src, cv::Mat& dst);
int blur5x5_alternate(cv::Mat& src, cv::Mat& dst);
int sobelX3x3(cv::Mat& src, cv::Mat& dst);
int sobelY3x3(cv::Mat& src, cv::Mat& dst);
int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst);
int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels);
int detectFaces(cv::Mat& grey, std::vector<cv::Rect>& faces);
int drawBoxes(cv::Mat& frame, std::vector<cv::Rect>& faces, int minWidth, float scale);
int cartoon(cv::Mat& src, cv::Mat& dst, int levels, int magThreshold);
int negative(cv::Mat& src, cv::Mat& dst);
int colorizeFaces(cv::Mat& frame, cv::Mat& processedFrame, std::vector<cv::Rect>& faces);
int adjustBrightness(cv::Mat& src, cv::Mat& dst, int value);
int adjustContrast(cv::Mat& src, cv::Mat& dst, float alpha);
void drawHistogram(cv::Mat& src, cv::Mat& histImage);
int brightnessValue = 0;
float contrastValue = 1.0;


// Main function
int main(int argc, char* argv[]) {
    cv::VideoCapture* capdev;
    cv::Mat grayImagefunc;
    cv::Mat grayImage;
    cv::Mat processedFrame;
    char key1;
    cv::Mat magImage;
    cv::Mat blurQuantImage;
    cv::Mat sobelXdisplaysrc;
    cv::Mat sobelYdisplaysrc;
    cv::Mat gradsrc;
    cv::Mat negImage;
    cv::Mat histImage;
    std::vector<cv::Rect> faces;  // Declare faces vector
    int minWidth = 0;  // Set an appropriate value for minWidth
    float scale = 1.0;  // Set an appropriate value for scale
    cv::Mat cartoonImage;

    // open the video device
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }

    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
        (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame;
    grayImagefunc.create(refS, CV_8UC3);  // Create once outside the loop
    char key = -1;

    for (;;) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        // see if there is a waiting keystroke
        key = cv::waitKey(10);

        // Calling each functions using keypress
        if (key == '0' || key == -1)
        {
            cv::imshow("Video", frame);
        }
        else if (key != -1)
        {
            key1 = key;
        }
        // key b for blur 
        if (key1 == 'b') {
            
            blur5x5(frame, processedFrame);
            cv::imshow("Video", processedFrame);
            std::cout << "5x5 Blur filter applied\n";
            cv::waitKey(10);
        }
        // key n for applying alternate blur function
        if (key1 == 'n') {
            // Apply faster 5x5 blur filter
            blur5x5_alternate(frame, processedFrame);
            cv::imshow("Video", processedFrame);
            std::cout << "Faster 5x5 Blur filter applied\n";
        }
        //Key s to save an image
        if (key == 's')
        {
            if (cv::imwrite("savedImage.jpg", frame)) {
                printf("Successfully saved savedImage.jpg to the current directory\n");
            }
            else {
                printf("Error: Unable to save the image\n");
            }
        }
        // key q to quit 
        if (key == 'q') {
            break;
        }
        if (key1 == 'h')
        {// calling alternate greyscale function when we press the key h
            cv::cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);
            cv::imshow("Video", grayImage);
            std::cout << "Alternate Greyscale filter applied\n";
        }
        if (key1 == 'g')
        { // calling greyscale function when we press the key g
            grayImagefunc.create(frame.size(), frame.type());
            greyscale(frame, grayImagefunc);
            cv::imshow("Video", grayImagefunc);
            std::cout << "Greyscale filter applied\n";
        }
        if (key1 == 't')
        {
            // Apply sepia tone filter when we press t
            sepiaTone(frame, processedFrame);
            cv::imshow("Video", processedFrame);
            std::cout << "Sepia tone filter applied\n";
        }
        // key x for Sobel X
        if (key1 == 'x')
        {
            sobelX3x3(frame, gradsrc);
            cv::convertScaleAbs(gradsrc, sobelXdisplaysrc);
            cv::imshow("Video", sobelXdisplaysrc);
        }
        // Key y for sobel Y
        if (key1 == 'y')
        {
            sobelY3x3(frame, gradsrc);
            cv::convertScaleAbs(gradsrc, sobelYdisplaysrc);
            cv::imshow("Video", sobelYdisplaysrc);
        }
        // Key m for sobel magnitude
        if (key1 == 'm')
            {
                sobelX3x3(frame, gradsrc);
                cv::convertScaleAbs(gradsrc, sobelXdisplaysrc);
                sobelY3x3(frame, gradsrc);
                cv::convertScaleAbs(gradsrc, sobelYdisplaysrc);
                magImage.create(frame.size(), frame.type());
                magnitude(sobelXdisplaysrc, sobelYdisplaysrc, magImage);
                cv::imshow("Video", magImage);
            }
        // Key l for blur quantization
        if (key1 == 'l')
        {
            blurQuantImage.create(frame.size(), frame.type());
            blurQuantize(frame, blurQuantImage, 15);
            cv::imshow("Video", blurQuantImage);
        }

        if (key1 == 'f') {
            // Detect faces and draw boxes when we press f
            cv::cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);
            detectFaces(grayImage, faces);
            drawBoxes(frame, faces, minWidth, scale);
            cv::imshow("Video", frame);
            std::cout << "Face detection applied\n";
        }
        
        if (key1 == 'z') //  displaying the negative of an image
        {
            negImage.create(frame.size(), frame.type());
            negative(frame, negImage);
            cv::imshow("Video", negImage);
        }
       // Adjust brightness based on keys
        if (key == 'j') {
            // Increase brightness
            brightnessValue += 20;
            adjustBrightness(frame, processedFrame, brightnessValue);
            cv::imshow("Video", processedFrame);
            std::cout << "Increased brightness\n";
            cv::waitKey(1000);
        }
        if (key == 'k') {
            // Decrease brightness
            brightnessValue -= 20;
            adjustBrightness(frame, processedFrame, brightnessValue);
            cv::imshow("Video", processedFrame);
            std::cout << "Decreased brightness\n";
            cv::waitKey(10000);
        }
        if (key == 'o') {
            // Increase contrast
            contrastValue += 0.5;
            adjustContrast(frame, processedFrame, contrastValue);
            cv::imshow("Video", processedFrame);
            std::cout << "Increased contrast\n";
            cv::waitKey(10000);
        }
        if (key == 'p') {
            // Decrease contrast
            contrastValue -= 0.5;
            adjustContrast(frame, processedFrame, contrastValue);
            cv::imshow("Video", processedFrame);
            std::cout << "Decreased contrast\n";
            cv::waitKey(10000);
        }
        if (key == 'd') {
            // Histogram
            drawHistogram(processedFrame, histImage);
            imshow("Histogram", histImage);
        }
    }
    delete capdev;
    return(0);
}
