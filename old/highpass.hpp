// #ifndef HIGHPASS_HPP
// #define HIGHPASS_HPP

// #include <opencv2/opencv.hpp>

// using namespace std;
// using namespace cv;

// void applyHighPassFilter(const Mat& src, Mat& dst, int kernel_size);
// void applyHighPassPixel(const Mat& src, Mat& dst, int x, int y, int kernel_size);

// #endif

#ifndef HIGHPASS_HPP
#define HIGHPASS_HPP

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

enum PaddingType { ZERO, MIRROR, REPLICATE };

void applyHighPassFilter(const Mat& src, Mat& dst, int kernel_size, PaddingType pad);
void applyHighPassPixel(const Mat& src, Mat& dst, int x, int y, int kernel_size, PaddingType pad);

#endif
