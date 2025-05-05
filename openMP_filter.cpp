#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace cv;
using namespace std;

// Function to generate a dynamic high-pass filter kernel
vector<int> generateKernel(int kernelSize) {
    vector<int> kernel(kernelSize * kernelSize, -1);
    int center = kernelSize / 2;
    kernel[center * kernelSize + center] = kernelSize * kernelSize - 1; // Center weight
    return kernel;
}

// Function to apply a high-pass filter using OpenMP
void applyHighPassFilter(const unsigned char* imageArray, unsigned char* filteredImage, int rows, int cols, const vector<int>& kernel, int kernelSize) {
    int offset = kernelSize / 2;

    #pragma omp parallel for schedule(dynamic)
    for (int i = offset; i < rows - offset; ++i) {
        for (int j = offset; j < cols - offset; ++j) {
            int sum = 0;
            for (int k = -offset; k <= offset; ++k) {
                for (int l = -offset; l <= offset; ++l) {
                    sum += imageArray[(i + k) * cols + (j + l)] * kernel[(k + offset) * kernelSize + (l + offset)];
                }
            }
            filteredImage[i * cols + j] = static_cast<unsigned char>(std::min(std::max(sum, 0), 255));
        }
    }
}

int main() {
    // Read image in grayscale
    cv::Mat img = cv::imread("Input/lena.png", cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cerr << "Error: Could not open or find the image.\n";
        return 1;
    }
    cout << "Image Loaded Successfully\n";

    int rows = img.rows;
    int cols = img.cols;
    std::cout << "Image dimensions: " << rows << " x " << cols << "\n";

    // Flatten the image into a 1D array for better memory locality
    unsigned char* imageArray = img.data;

    while (true) {
        // Get kernel size from the user
        int kernelSize;
        cout << "Enter the kernel size (odd number greater than 1, or 0 to exit): ";
        cin >> kernelSize;

        if (kernelSize == 0) {
            cout << "Exiting program.\n";
            break;
        }

        if (kernelSize < 3 || kernelSize % 2 == 0) {
            cerr << "Error: Kernel size must be an odd number greater than or equal to 3.\n";
            continue;
        }

        // Generate the kernel
        vector<int> kernel = generateKernel(kernelSize);

        // Allocate memory for the filtered image
        unsigned char* filteredImage = new unsigned char[rows * cols]();

        // Start measuring time
        auto start = std::chrono::high_resolution_clock::now();

        // Apply high-pass filter
        applyHighPassFilter(imageArray, filteredImage, rows, cols, kernel, kernelSize);

        // Stop measuring time
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        // Convert the filtered image back to cv::Mat
        cv::Mat filteredImg(rows, cols, CV_8UC1, filteredImage);

        // Display the filtered image
        cv::imshow("Filtered Image", filteredImg);
        cv::waitKey(0);
        cv::destroyAllWindows();

        // Save the filtered image
        string outputFileName = "Output/openMP/kernel_" + to_string(kernelSize) + ".jpg";
        cv::imwrite(outputFileName, filteredImg);
        cout << "Filtered image with kernel size " << kernelSize << " saved successfully as " << outputFileName << ".\n";

        // Print the elapsed time
        std::cout << "Execution time for kernel size " << kernelSize << ": " << elapsed.count() << " seconds.\n";

        // Free allocated memory
        delete[] filteredImage;
    }

    return 0;
}