#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <mpi.h>
#include <chrono>

using namespace cv;
using namespace std;

// Function to generate a dynamic high-pass filter kernel
vector<int> generateKernel(int kernelSize) {
    vector<int> kernel(kernelSize * kernelSize, -1);
    int center = kernelSize / 2;
    kernel[center * kernelSize + center] = kernelSize * kernelSize - 1; // Center weight
    return kernel;
}

// Function to apply a high-pass filter to a chunk of the image
void applyHighPassFilter(const unsigned char* imageArray, unsigned char* filteredImage, int rows, int cols, const vector<int>& kernel, int kernelSize, int startRow, int endRow) {
    int offset = kernelSize / 2;

    for (int i = startRow; i < endRow; ++i) {
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cv::Mat img;
    int rows, cols;

    if (rank == 0) {
        // Process 0 reads the image
        img = cv::imread("Input/lena.png", cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Error: Could not open or find the image.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        cout << "Image Loaded Successfully\n";

        rows = img.rows;
        cols = img.cols;
    }

    // Broadcast image dimensions to all processes
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for the image and filtered image
    unsigned char* imageArray = nullptr;
    unsigned char* filteredImage = new unsigned char[rows * cols]();

    if (rank == 0) {
        imageArray = img.data;
    }

    while (true) {
        int kernelSize;
        vector<int> kernel;

        if (rank == 0) {
            // Prompt the user for the kernel size
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

            kernel = generateKernel(kernelSize);
        }

        // Broadcast the kernel size and kernel to all processes
        MPI_Bcast(&kernelSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            kernel.resize(kernelSize * kernelSize);
        }
        MPI_Bcast(kernel.data(), kernel.size(), MPI_INT, 0, MPI_COMM_WORLD);

        // Calculate sendcounts and displacements for MPI_Scatterv
        vector<int> sendcounts(size, 0);
        vector<int> displs(size, 0);

        int rowsPerProcess = rows / size;
        int extraRows = rows % size;

        for (int i = 0; i < size; ++i) {
            sendcounts[i] = (rowsPerProcess + (i < extraRows ? 1 : 0)) * cols;
            displs[i] = (i > 0 ? displs[i - 1] + sendcounts[i - 1] : 0);
        }

        int localRows = sendcounts[rank] / cols;
        unsigned char* localImageArray = new unsigned char[sendcounts[rank] + 2 * cols]; // Add halo rows
        unsigned char* localFilteredImage = new unsigned char[sendcounts[rank]]();

        // Scatter rows of the image to all processes
        MPI_Scatterv(imageArray, sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR, localImageArray + cols, sendcounts[rank], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        // Copy halo rows for boundary handling
        if (rank > 0) {
            MPI_Recv(localImageArray, cols, MPI_UNSIGNED_CHAR, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Send(localImageArray + sendcounts[rank], cols, MPI_UNSIGNED_CHAR, rank + 1, 0, MPI_COMM_WORLD);
        }

        // Measure the time taken by the applyHighPassFilter function
        auto start = std::chrono::high_resolution_clock::now();

        // Apply the high-pass filter to the local chunk
        int startRow = displs[rank] / cols;
        int endRow = startRow + localRows;
        applyHighPassFilter(localImageArray, localFilteredImage, rows, cols, kernel, kernelSize, startRow, endRow);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        if (rank == 0) {
            cout << "Time taken by applyHighPassFilter for kernel size " << kernelSize << ": " << elapsed.count() << " seconds.\n";
        }

        // Gather the filtered chunks back to process 0
        MPI_Gatherv(localFilteredImage, sendcounts[rank], MPI_UNSIGNED_CHAR, filteredImage, sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Convert the filtered image back to cv::Mat
            cv::Mat filteredImg(rows, cols, CV_8UC1, filteredImage);

            // Save and display the filtered image
            string outputFileName = "Output/MPI/kernel_" + to_string(kernelSize) + ".jpg";
            cv::imwrite(outputFileName, filteredImg);
            cout << "Filtered image saved successfully as " << outputFileName << ".\n";

            cv::imshow("Filtered Image", filteredImg);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }

        // Free allocated memory for this iteration
        delete[] localImageArray;
        delete[] localFilteredImage;
    }

    // Free allocated memory
    delete[] filteredImage;

    MPI_Finalize();
    return 0;
}