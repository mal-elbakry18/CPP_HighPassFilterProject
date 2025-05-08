// #include "highpass.hpp"

// void applyHighPassPixel(const Mat& src, Mat& dst, int x, int y, int kernel_size) {
//     int k = kernel_size / 2;
//     for (int c = 0; c < 3; ++c) {
//         int sum = 0;
//         int count = 0;

//         for (int i = -k; i <= k; ++i) {
//             for (int j = -k; j <= k; ++j) {
//                 int newY = y + i;
//                 int newX = x + j;

//                 // Mirror padding
//                 newY = std::min(std::max(newY, 0), src.rows - 1);
//                 newX = std::min(std::max(newX, 0), src.cols - 1);

//                 sum += src.at<Vec3b>(newY, newX)[c];
//                 count++;
//             }
//         }

//         int mean = sum / count;
//         int highpass = src.at<Vec3b>(y, x)[c] - mean;
//         dst.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(highpass);
//     }
// }

// void applyHighPassFilter(const Mat& src, Mat& dst, int kernel_size) {
//     dst = src.clone();
//     int k = kernel_size / 2;

//     for (int y = 0; y < src.rows; ++y) {
//         for (int x = 0; x < src.cols; ++x) {
//             applyHighPassPixel(src, dst, x, y, kernel_size);
//         }
//     }
// }


#include "highpass.hpp"

void applyHighPassPixel(const Mat& src, Mat& dst, int x, int y, int kernel_size, PaddingType pad) {
    int k = kernel_size / 2;

    for (int c = 0; c < 3; ++c) {
        int sum = 0;
        int count = 0;

        for (int i = -k; i <= k; ++i) {
            for (int j = -k; j <= k; ++j) {
                int newY = y + i;
                int newX = x + j;

                int value;
                if (newY < 0 || newY >= src.rows || newX < 0 || newX >= src.cols) {
                    // Out of bounds handling
                    if (pad == ZERO) {
                        value = 0;
                    } else if (pad == MIRROR) {
                        newY = std::min(std::max(newY, 0), src.rows - 1);
                        newX = std::min(std::max(newX, 0), src.cols - 1);
                        value = src.at<Vec3b>(newY, newX)[c];
                    } else if (pad == REPLICATE) {
                        newY = (newY < 0) ? 0 : (newY >= src.rows ? src.rows - 1 : newY);
                        newX = (newX < 0) ? 0 : (newX >= src.cols ? src.cols - 1 : newX);
                        value = src.at<Vec3b>(newY, newX)[c];
                    }
                } else {
                    value = src.at<Vec3b>(newY, newX)[c];
                }

                sum += value;
                count++;
            }
        }

        int mean = sum / count;
        int highpass = src.at<Vec3b>(y, x)[c] - mean;
        dst.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(highpass);
    }
}

void applyHighPassFilter(const Mat& src, Mat& dst, int kernel_size, PaddingType pad) {
    dst = src.clone();
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            applyHighPassPixel(src, dst, x, y, kernel_size, pad);
        }
    }
}
