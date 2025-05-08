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

// Function to generate a dynamic high-pass filter kernel
vector<vector<int>> generateKernel(int kernelSize) {
    vector<vector<int>> kernel(kernelSize, vector<int>(kernelSize, -1));
    int center = kernelSize / 2;
    kernel[center][center] = kernelSize * kernelSize - 1; // Center weight
    return kernel;
}

// Function to apply a high-pass filter
vector<vector<int>> applyHighPassFilter(const vector<vector<unsigned char>>& imageArray, int kernelSize) {
    int rows = imageArray.size();
    int cols = imageArray[0].size();
    vector<vector<int>> filteredImage(rows, vector<int>(cols, 0));

    // Generate the kernel dynamically
    vector<vector<int>> kernel = generateKernel(kernelSize);
    int offset = kernelSize / 2;

    for (int i = offset; i < rows - offset; ++i) {
        for (int j = offset; j < cols - offset; ++j) {
            int sum = 0;
            for (int k = -offset; k <= offset; ++k) {
                for (int l = -offset; l <= offset; ++l) {
                    sum += imageArray[i + k][j + l] * kernel[k + offset][l + offset];
                }
            }
            filteredImage[i][j] = std::min(std::max(sum, 0), 255); // Clamp to [0, 255]
        }
    }

    return filteredImage;
}

int main(int argc, char** argv) {
    // // Read image in grayscale
    // cv::Mat img = cv::imread("Input/lena.png", cv::IMREAD_GRAYSCALE);

    // if (img.empty()) {
    //     std::cerr << "Error: Could not open or find the image.\n";
    //     return 1;
    // }
    // cout << "Image Loaded Successfully\n";
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <image_path> <kernel_size>\n";
        return 1;
    }

    string imagePath = argv[1];
    int kernelSize = atoi(argv[2]);

    if (kernelSize < 3 || kernelSize % 2 == 0) {
        cerr << "Error: Kernel size must be an odd number greater than or equal to 3.\n";
        return 1;
    }

    // Read image in grayscale
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cerr << "Error: Could not open or find the image at " << imagePath << "\n";
        return 1;
    }
    cout << "Image Loaded Successfully\n";

    // Convert image to 2D array
    vector<vector<unsigned char>> imageArray = imageTo2DArray(img);
    cout << "Image converted to 2D array successfully.\n";

    int rows = img.rows;
    int cols = img.cols;
    std::cout << "Image dimensions: " << rows << " x " << cols << "\n";

    // while (true) {
    //     // Get kernel size from the user
    //     int kernelSize;
    //     cout << "Enter the kernel size (odd number greater than 1, or 0 to exit): ";
    //     cin >> kernelSize;

    //     if (kernelSize == 0) {
    //         cout << "Exiting program.\n";
    //         break;
    //     }

    //     if (kernelSize < 3 || kernelSize % 2 == 0) {
    //         cerr << "Error: Kernel size must be an odd number greater than or equal to 3.\n";
    //         continue;
    //     }

    //     // Start measuring time
    //     auto start = std::chrono::high_resolution_clock::now();

    //     // Apply high-pass filter with the user-defined kernel size
    //     vector<vector<int>> filteredImage = applyHighPassFilter(imageArray, kernelSize);

    //     auto end = std::chrono::high_resolution_clock::now();
        
    //     // Display the filtered image
    //     cv::Mat filteredImg(rows, cols, CV_8UC1);
    //     for (int i = 0; i < rows; ++i) {
    //         for (int j = 0; j < cols; ++j) {
    //             filteredImg.at<uchar>(i, j) = static_cast<uchar>(filteredImage[i][j]);
    //         }
    //     }
    //     cv::imshow("Filtered Image", filteredImg);
    //     // cv::waitKey(0);
    //     cv::destroyAllWindows();

    //     // Save the filtered image
    //     string outputFileName = "Output/sequential/kernel_" + to_string(kernelSize) + ".jpg";
    //     cv::imwrite(outputFileName, filteredImg);
    //     cout << "Filtered image with kernel size " << kernelSize << " saved successfully as " << outputFileName << ".\n";

    //     // Stop measuring time
    //     std::chrono::duration<double> elapsed = end - start;

    //     // Print the elapsed time
    //     std::cout << "Execution time for kernel size " << kernelSize << ": " << elapsed.count() << " seconds.\n";
    // }

    // Start measuring time
    auto start = std::chrono::high_resolution_clock::now();

    // Apply high-pass filter with the provided kernel size
    vector<vector<int>> filteredImage = applyHighPassFilter(imageArray, kernelSize);

    auto end = std::chrono::high_resolution_clock::now();

    // // Display the filtered image
    // cv::Mat filteredImg(rows, cols, CV_8UC1);
    // for (int i = 0; i < rows; ++i) {
    //     for (int j = 0; j < cols; ++j) {
    //         filteredImg.at<uchar>(i, j) = static_cast<uchar>(filteredImage[i][j]);
    //     }
    // }
    // cv::imshow("Filtered Image", filteredImg);
    // cv::waitKey(0);
    // cv::destroyAllWindows();
    
    cv::Mat filteredImg(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            filteredImg.at<uchar>(i, j) = static_cast<uchar>(filteredImage[i][j]);
        }
    }

    // Save the filtered image
    string outputFileName = "Output/sequential/kernel_" + to_string(kernelSize) + ".jpg";
    cv::imwrite(outputFileName, filteredImg);
    cout << "Filtered image with kernel size " << kernelSize << " saved successfully as " << outputFileName << ".\n";

    // Stop measuring time
    std::chrono::duration<double> elapsed = end - start;

    // Print the elapsed time
    std::cout << "Execution time for kernel size " << kernelSize << ": " << elapsed.count() << " seconds.\n";


    return 0;
}