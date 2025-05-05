#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using namespace cv;

std::chrono::time_point<std::chrono::high_resolution_clock> start;

// 3x3 Laplacian kernel
int kernel[3][3] = {
    {  0, -1,  0 },
    { -1,  4, -1 },
    {  0, -1,  0 }
};

uchar apply_kernel(const Mat& img, int row, int col) {
    int sum = 0;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            int r = min(max(row + i, 0), img.rows - 1);
            int c = min(max(col + j, 0), img.cols - 1);
            sum += img.at<uchar>(r, c) * kernel[i + 1][j + 1];
        }
    }
    return saturate_cast<uchar>(sum);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Mat image, gray;
    int rows, cols;

    if (rank == 0) {
        image = imread("Input/lena.png", IMREAD_GRAYSCALE);
        if (image.empty()) {
            cerr << "Image not found!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        rows = image.rows;
        cols = image.cols;
        cout << "Image Loaded Successfully" << endl;
        cout << "Image dimensions: " << rows << " x " << cols << endl;
        start = chrono::high_resolution_clock::now();
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_proc = rows / size;
    int remainder = rows % size;
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

    // Allocate local buffer with padding for filtering
    Mat local_chunk(local_rows + 2, cols, CV_8UC1, Scalar(0));

    vector<int> sendcounts(size), displs(size);
    int offset = 0;
    for (int i = 0; i < size; ++i) {
        int rpp = rows_per_proc + (i < remainder ? 1 : 0);
        sendcounts[i] = rpp * cols;
        displs[i] = offset;
        offset += rpp * cols;
    }

    if (rank == 0) {
        // Prepare data for scatter
        vector<uchar> flat_image(image.begin<uchar>(), image.end<uchar>());
        MPI_Scatterv(flat_image.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
                     local_chunk.ptr(1), local_rows * cols, MPI_UNSIGNED_CHAR,
                     0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_UNSIGNED_CHAR,
                     local_chunk.ptr(1), local_rows * cols, MPI_UNSIGNED_CHAR,
                     0, MPI_COMM_WORLD);
    }

    // Exchange rows with neighbors
    if (rank > 0)
        MPI_Sendrecv(local_chunk.ptr(1), cols, MPI_UNSIGNED_CHAR, rank - 1, 0,
                     local_chunk.ptr(0), cols, MPI_UNSIGNED_CHAR, rank - 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (rank < size - 1)
        MPI_Sendrecv(local_chunk.ptr(local_rows), cols, MPI_UNSIGNED_CHAR, rank + 1, 0,
                     local_chunk.ptr(local_rows + 1), cols, MPI_UNSIGNED_CHAR, rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Apply filter
    Mat filtered(local_rows, cols, CV_8UC1);
    for (int i = 1; i <= local_rows; ++i)
        for (int j = 0; j < cols; ++j)
            filtered.at<uchar>(i - 1, j) = apply_kernel(local_chunk, i, j);

    // Gather result
    Mat final_image;
    if (rank == 0)
        final_image = Mat(rows, cols, CV_8UC1);

    MPI_Gatherv(filtered.data, local_rows * cols, MPI_UNSIGNED_CHAR,
                final_image.data, sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        cout << "Time taken: " << elapsed.count() << " seconds" << endl;
        imwrite("output.jpg", final_image);
        cout << "High-pass filter applied and saved to output.jpg" << endl;
    }

    MPI_Finalize();
    return 0;
}
