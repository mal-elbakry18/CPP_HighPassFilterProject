#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>

using namespace cv;
using namespace std;

// Function to convert an image to a 2D array
vector<vector<unsigned char>> imageTo2DArray(const Mat& img) {
    int rows = img.rows;
    int cols = img.cols;
    vector<vector<unsigned char>> array(rows, vector<unsigned char>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            array[i][j] = img.at<uchar>(i, j); // Grayscale pixel
        }
    }
    return array;
}

vector<vector<int>> applyHighPassFilter(const vector<vector<unsigned char>>& imageArray, int kernelSize = 5) {
    int rows = imageArray.size();
    int cols = imageArray[0].size();
    vector<vector<int>> filteredImage(rows, vector<int>(cols, 0));

    // High-pass filter kernel
    int kernel1[5][5] = {
        {  0,  0, -1,  0,  0 },
        {  0, -1, -2, -1,  0 },
        { -1, -2, 16, -2, -1 },
        {  0, -1, -2, -1,  0 },
        {  0,  0, -1,  0,  0 }
    };

    int kernel2[3][3] = {
        { 0, -1, 0 },
        { -1, 4, -1 },
        { 0, -1, 0 }
    };

    switch (kernelSize) {
        case 3: // 3x3 kernel
            // Apply the 3x3 kernel
            for (int i = 1; i < rows - 1; ++i) {
                for (int j = 1; j < cols - 1; ++j) {
                    int sum = 0;
                    for (int k = -1; k <= 1; ++k) {
                        for (int l = -1; l <= 1; ++l) {
                            sum += imageArray[i + k][j + l] * kernel2[k + 1][l + 1];
                        }
                    }
                    filteredImage[i][j] = std::min(std::max(sum, 0), 255); // Clamp to [0, 255]
                }
            }
            break;
        case 5: // 5x5 kernel
            // Apply the 5x5 kernel
            for (int i = 2; i < rows - 2; ++i) {
                for (int j = 2; j < cols - 2; ++j) {
                    int sum = 0;
                    for (int k = -2; k <= 2; ++k) {
                        for (int l = -2; l <= 2; ++l) {
                            sum += imageArray[i + k][j + l] * kernel1[k + 2][l + 2];
                        }
                    }
                    filteredImage[i][j] = std::min(std::max(sum, 0), 255); // Clamp to [0, 255]
                }
            }
            break;
        default:
            std::cerr << "Error: Unsupported kernel size. Use 3 or 5.\n";
            return filteredImage; // Return empty image
    }
    
    return filteredImage;
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    // Read image in grayscale
    cv::Mat img = cv::imread("Input/lena.png", cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cerr << "Error: Could not open or find the image.\n";
        return 1;
    }
    cout << "Image Loaded Successfully\n";
    // Convert image to 2D array

    vector<vector<unsigned char>> imageArray = imageTo2DArray(img);
    cout << "Image converted to 2D array successfully.\n";

    int rows = img.rows;
    int cols = img.cols;
    std::cout << "Image dimensions: " << rows << " x " << cols << "\n";

    // Apply high-pass filter with 5x5 kernel
    vector<vector<int>> filteredImage = applyHighPassFilter(imageArray, 5);
    // display the filtered image
    cv::Mat filteredImg(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            filteredImg.at<uchar>(i, j) = static_cast<uchar>(filteredImage[i][j]);
        }
    }
    cv::imshow("Filtered Image", filteredImg);
    // cv::waitKey(0);
    cv::destroyAllWindows();
    // Save the filtered image
    cv::imwrite("Output/sequential/5x5.jpg", filteredImg);

    cout << "Filtered image saved successfully.\n";

    // Apply high-pass filter with 3x3 kernel
    cout << "Applying 3x3 kernel...\n";
    vector<vector<int>> filteredImage3x3 = applyHighPassFilter(imageArray, 3);
    // display the filtered image
    cv::Mat filteredImg3x3(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            filteredImg3x3.at<uchar>(i, j) = static_cast<uchar>(filteredImage3x3[i][j]);
        }
    }
    cv::imshow("Filtered Image 3x3", filteredImg3x3);
    // cv::waitKey(0); 
    cv::destroyAllWindows();
    // Save the filtered image
    cv::imwrite("Output/sequential/3x3.jpg", filteredImg3x3);
    cout << "Filtered image with 3x3 kernel saved successfully.\n";

    std::cout << "Image converted to 2D array successfully.\n";

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Print the elapsed time
    std::cout << "Total execution time: " << elapsed.count() << " seconds.\n";
    return 0;
}
