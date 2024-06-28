/*
Basil Reji & Kevin Sebastian Sani
Spring 2024
Pattern Recognition & Computer Vision
Assignment 1 - Video Special Effects
Filter.cpp file containing all functions relating to video manipulation
*/

#include<opencv2/opencv.hpp>
#include <iostream>
#include<math.h>
#include "filter.h"
#include <ctime>
#include <cmath>
#include "faceDetect.h"
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

// Task 3: Function for converting a color video feed into greyscale

int greyscale(cv::Mat& src, cv::Mat& dst)
{
    for (int i = 0; i < src.rows; ++i)
    {
        cv::Vec3b* rptrs = src.ptr <cv::Vec3b>(i);
        cv::Vec3b* rptrd = dst.ptr <cv::Vec3b>(i);
        for (int j = 0; j < src.cols; ++j)
        {
            uchar avg = (rptrs[j][0] + rptrs[j][1] + rptrs[j][2]) / 3;
            rptrd[j][0] = avg;
            rptrd[j][1] = avg;
            rptrd[j][2] = avg;
        }
    }
    return 0;
}

//Task 5: Function for Sepia Tone Filter
int sepiaTone(cv::Mat& src, cv::Mat& dst)
{
    dst.create(src.size(), src.type());
    for (int i = 0; i < src.rows; ++i)
    {
        cv::Vec3b* rptrs = src.ptr <cv::Vec3b>(i);
        cv::Vec3b* rptrd = dst.ptr <cv::Vec3b>(i);

        for (int j = 0; j < src.cols; ++j)
        {
            // Store original RGB values
            uchar R = rptrs[j][2];
            uchar G = rptrs[j][1];
            uchar B = rptrs[j][0];

            // Calculate new RGB values using the Sepia tone matrix
            rptrd[j][2] = cv::saturate_cast<uchar>(0.272 * R + 0.534 * G + 0.131 * B);
            rptrd[j][1] = cv::saturate_cast<uchar>(0.349 * R + 0.686 * G + 0.168 * B);
            rptrd[j][0] = cv::saturate_cast<uchar>(0.393 * R + 0.769 * G + 0.189 * B);
        }
    }
    return 0;
}
//Task 6: Function for 5x5 blur
int blur5x5(cv::Mat& src, cv::Mat& dst) {

    // 5x5 blur filter implementation 
    dst.create(src.size(), src.type());
    // Add timing code
    clock_t start = clock();
    // Your blur implementation

    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    std::cout << "Time taken by function: " << cpu_time_used << " seconds\n";

    return 0;
}
// Task 5b: Alternate method to do 5x5 blur
int blur5x5_alternate(cv::Mat& src, cv::Mat& dst) {
    // Create a copy of the source image
    dst = src.clone();
    // Define 1x5 separable filters
    float filter[5] = { 1, 2, 4, 2, 1 };
    // Create vertical filter matrix
    cv::Mat verticalFilter(5, 1, CV_32F, filter);
    // Create horizontal filter matrix
    cv::Mat horizontalFilter(1, 5, CV_32F, filter);
    // Add timing code
    clock_t start = clock();
    // Apply vertical filter
    cv::sepFilter2D(src, dst, -1, verticalFilter, cv::Mat(), cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    // Apply horizontal filter
    cv::sepFilter2D(dst, dst, -1, horizontalFilter, cv::Mat(), cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    std::cout << "Time taken by blur5x5_2: " << cpu_time_used << " seconds\n";
    return 0;
}

// Task 7a: Function for Sobel X Filter
int sobelX3x3(cv::Mat& src, cv::Mat& dst)
{
    cv::Mat temp = cv::Mat::zeros(src.size(), CV_16SC3);
    cv::Mat temp2 = cv::Mat::zeros(src.size(), CV_16SC3);

    // Horizontal filter
    for (int i = 1; i < src.rows - 1; ++i)
    {
        cv::Vec3b* rptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3s* dptr = temp.ptr<cv::Vec3s>(i);

        for (int j = 1; j < src.cols - 1; ++j)
        {
            for (int c = 0; c <= 2; c++)
            {
                dptr[j][c] = (-1 * rptr[j - 1][c] + rptr[j + 1][c]) / 2;
            }
        }
    }

    // Vertical filter
    for (int i = 1; i < src.rows - 1; ++i)
    {
        cv::Vec3s* rptr = temp.ptr<cv::Vec3s>(i);
        cv::Vec3s* dptr = temp2.ptr<cv::Vec3s>(i);

        for (int j = 1; j < src.cols - 1; ++j)
        {
            for (int c = 0; c <= 2; c++)
            {
                dptr[j][c] = (-1 * rptr[j - 1][c] + rptr[j + 1][c]) / 2;
            }
        }
    }

    temp2.copyTo(dst);
    return 0;
}


// Task 7b: Function For the 3x3 Sobel Y filter

int sobelY3x3(cv::Mat& src, cv::Mat& dst)
{
    cv::Mat temp = cv::Mat::zeros(src.size(), CV_16SC3);

    for (int i = 1; i < src.rows - 1; ++i)
    {
        cv::Vec3b* rptrm1 = src.ptr <cv::Vec3b>(i - 1);
        cv::Vec3b* rptr = src.ptr <cv::Vec3b>(i);
        cv::Vec3b* rptrp1 = src.ptr <cv::Vec3b>(i + 1);

        cv::Vec3s* dptr = temp.ptr <cv::Vec3s>(i);
        for (int j = 1; j < src.cols - 1; ++j)
        {
            for (int c = 0; c <= 2; c++)
            {
                dptr[j][c] = (1 * rptrm1[j - 1][c] + 2 * rptrm1[j][c] + rptrm1[j + 1][c]
                    - 1 * rptrp1[j - 1][c] - 2 * rptrp1[j][c] - rptrp1[j + 1][c]) / 4;
            }
        }
    }

    temp.copyTo(dst);
    return 0;

}

// Task 8: Function for Gradient magnitude image from sobel X and sobel Y
int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst)
{
    dst = cv::Mat::zeros(sx.size(), sx.type());
    for (int i = 0; i < sx.rows; ++i)
    {
        cv::Vec3b* rptrsx = sx.ptr <cv::Vec3b>(i);
        cv::Vec3b* rptrsy = sy.ptr <cv::Vec3b>(i);
        cv::Vec3b* rptrd = dst.ptr <cv::Vec3b>(i);
        for (int j = 0; j < sx.cols; ++j)
        {
            rptrd[j][0] = sqrt((rptrsx[j][0] * rptrsx[j][0]) + (rptrsy[j][0] * rptrsy[j][0]));
            rptrd[j][1] = sqrt((rptrsx[j][1] * rptrsx[j][1]) + (rptrsy[j][1] * rptrsy[j][1]));
            rptrd[j][2] = sqrt((rptrsx[j][2] * rptrsx[j][2]) + (rptrsy[j][2] * rptrsy[j][2]));
        }
    }
    return 0;
}

//Task 9: Function for blur quantization
int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels)
{
    blur5x5(src, dst);
    int bucket = 255 / levels;
    for (int i = 0; i < src.rows; ++i)
    {
        cv::Vec3b* rptrd = dst.ptr <cv::Vec3b>(i);
        for (int j = 1; j < src.cols; ++j)
        {
            rptrd[j][0] = (rptrd[j][0] / bucket) * bucket;
            rptrd[j][1] = (rptrd[j][1] / bucket) * bucket;
            rptrd[j][2] = (rptrd[j][2] / bucket) * bucket;
        }
    }
    return 0;
}

//Task 10: Function to Detect face
int detectFaces(cv::Mat& grey, std::vector<cv::Rect>& faces) {
    // a static variable to hold a half-size image
    static cv::Mat half;
    // a static variable to hold the classifier
    static cv::CascadeClassifier face_cascade;
    // the path to the haar cascade file
    static cv::String face_cascade_file(FACE_CASCADE_FILE);

    if (face_cascade.empty()) {
        if (!face_cascade.load(face_cascade_file)) {
            printf("Unable to load face cascade file\n");
            printf("Terminating\n");
            exit(-1);
        }
    }
    // clear the vector of faces
    faces.clear();
    // cut the image size in half to reduce processing time
    cv::resize(grey, half, cv::Size(grey.cols / 2, grey.rows / 2));
    // equalize the image
    cv::equalizeHist(half, half);
    // apply the Haar cascade detector
    face_cascade.detectMultiScale(half, faces);
    // adjust the rectangle sizes back to the full size image
    for (int i = 0; i < faces.size(); i++) {
        faces[i].x *= 2;
        faces[i].y *= 2;
        faces[i].width *= 2;
        faces[i].height *= 2;
    }

    return(0);
}

/* ------------------------    Task 11 Special Effects on the Video    ------------------------*/


// Function to draw boxes on the faces when faces are detected
int drawBoxes(cv::Mat& frame, std::vector<cv::Rect>& faces, int minWidth, float scale) {
    // The color to draw, you can change it here (B, G, R)
    cv::Scalar wcolor(170, 120, 110);

    for (int i = 0; i < faces.size(); i++) {
        if (faces[i].width > minWidth) {
            cv::Rect face(faces[i]);
            face.x *= scale;
            face.y *= scale;
            face.width *= scale;
            face.height *= scale;
            cv::rectangle(frame, face, wcolor, 3);
        }
    }

    return(0);
}

// Making a negative of the image
int negative(cv::Mat& src, cv::Mat& dst)
{
    dst = cv::Mat::zeros(src.size(), src.type());
    for (int i = 0; i < src.rows; ++i)
    {
        cv::Vec3b* rptrs = src.ptr <cv::Vec3b>(i);
        cv::Vec3b* rptrd = dst.ptr <cv::Vec3b>(i);
        for (int j = 1; j < src.cols; ++j)
        {
            rptrd[j][0] = 255 - rptrs[j][0];
            rptrd[j][1] = 255 - rptrs[j][1];
            rptrd[j][2] = 255 - rptrs[j][2];
        }
    }
    return 0;
}

// Function to make face colorful while the rest of the image is greyscale
int colorizeFaces(cv::Mat& frame, cv::Mat& processedFrame, std::vector<cv::Rect>& faces) {
    processedFrame = frame.clone();  // Copy the input frame to the processedFrame

    // Colorize each detected face in the processedFrame
    for (const auto& face : faces) {
        cv::Mat roi = processedFrame(face);
        cv::cvtColor(roi, roi, cv::COLOR_BGR2HSV);
        // Modify the hue, saturation, and value as needed
        // For example, you can set the hue to a specific color
        roi.setTo(cv::Scalar(120, 255, 255));
        cv::cvtColor(roi, roi, cv::COLOR_HSV2BGR);
    }

    return 0;
}
// Function to adjust brightness
int adjustBrightness(cv::Mat& src, cv::Mat& dst, int value) {
    src.convertTo(dst, -1, 1, value);
    return 0;
}

// Function to adjust contrast
int adjustContrast(cv::Mat& src, cv::Mat& dst, float alpha) {
    src.convertTo(dst, -1, alpha, 0);
    return 0;
}

