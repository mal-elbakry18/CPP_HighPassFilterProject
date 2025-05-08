// #include <mpi.h>
// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <vector>
// #include <chrono>

// using namespace std;
// using namespace cv;

// //CV_8UC1 --> 8 bit unsigned integer and c1 is one channel ( as it is greyscale)

// std::chrono::time_point<std::chrono::high_resolution_clock> start;

// // // 3x3 Laplacian kernel
// // int kernel[3][3] = {
// //     {  0, -1,  0 },
// //     { -1,  4, -1 },
// //     {  0, -1,  0 }
// // };


// vector<vector<int>> generateKernel(int ksize) {
//     vector<vector<int>> kernel(ksize, vector<int>(ksize, -1));
//     kernel[ksize / 2][ksize / 2] = ksize * ksize - 1;
//     return kernel;
// }

// uchar apply_kernel(const Mat& img, int row, int col, const vector<vector<int>>& kernel, int ksize) {
//     int sum = 0;
//     int offset = ksize / 2;
//     for (int i = -offset; i <= offset; ++i) {
//         for (int j = -offset; j <= offset; ++j) {
//             int r = min(max(row + i, 0), img.rows - 1);
//             int c = min(max(col + j, 0), img.cols - 1);
//             sum += img.at<uchar>(r, c) * kernel[i + offset][j + offset];
//         }
//     }
//     return saturate_cast<uchar>(sum);
// }


// int main(int argc, char** argv) {
//     MPI_Init(&argc, &argv);

//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);


//     if (argc < 3) {
//         if (rank == 0)
//             cerr << "Usage: mpirun -np <num_processes> ./MPI_filter <image_path> <kernel_size>" << endl;
//         MPI_Finalize();
//         return 1;
//     }
    
//     string image_path = argv[1];
//     int ksize = atoi(argv[2]);
//     vector<vector<int>> kernel = generateKernel(ksize);

    
//     if (ksize < 3 || ksize % 2 == 0) {
//         if (rank == 0)
//             cerr << "Kernel size must be odd and >= 3." << endl;
//         MPI_Finalize();
//         return 1;
//     }
    

//     Mat image, gray;
//     int rows, cols;

//     if (rank == 0) {
//         image = imread(image_path, IMREAD_GRAYSCALE);
//         if (image.empty()) {
//             cerr << "Image not found!" << endl;
//             MPI_Abort(MPI_COMM_WORLD, 1);
//         }
//         rows = image.rows;
//         cols = image.cols;
//         cout << "Image Loaded Successfully" << endl;
//         cout << "Image dimensions: " << rows << " x " << cols << endl;
//         start = chrono::high_resolution_clock::now();
//     }

//     MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     //Rows are divided among processes
//     //Some processes may get extra row for balancing
//     int rows_per_proc = rows / size;
//     int remainder = rows % size;
//     int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

//     // Allocate local buffer with padding for filtering
//     // Allocate 2 extra padding rows for safe convolution
//     // & these are exchanged with neighbors
//     Mat local_chunk(local_rows + 2, cols, CV_8UC1, Scalar(0));


//     //Calculate how many elements each process should receive and from where in the original image 
//     vector<int> sendcounts(size), displs(size);
//     int offset = 0;
//     for (int i = 0; i < size; ++i) {
//         int rpp = rows_per_proc + (i < remainder ? 1 : 0);
//         sendcounts[i] = rpp * cols;
//         displs[i] = offset;
//         offset += rpp * cols;
//     }

//     if (rank == 0) {
//         // Prepare data for scatter --> sent the internal chunk with specified start and end
//         // *With displacement *//
//         vector<uchar> flat_image(image.begin<uchar>(), image.end<uchar>());
//         MPI_Scatterv(flat_image.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
//                      local_chunk.ptr(1), local_rows * cols, MPI_UNSIGNED_CHAR,
//                      0, MPI_COMM_WORLD);
//     } else {
//         //only rank0 scatter and others also call it but pass null
//         //to avoid hang or crash of operation and to recieve their own chunk
//         MPI_Scatterv(nullptr, nullptr, nullptr, MPI_UNSIGNED_CHAR,
//                      local_chunk.ptr(1), local_rows * cols, MPI_UNSIGNED_CHAR,
//                      0, MPI_COMM_WORLD);
//     }

//     // Exchange rows with neighbors
//     // Each process exchange edge rows with neighbors to make the filtering accurate at the boundries
//     if (rank > 0)
//         MPI_Sendrecv(local_chunk.ptr(1), cols, MPI_UNSIGNED_CHAR, rank - 1, 0,
//                      local_chunk.ptr(0), cols, MPI_UNSIGNED_CHAR, rank - 1, 0,
//                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
//     //halo exchange / ghost rows between adjacent processes
//     //sending the last row to the next processes 
//     //receiving the first low from the next processes 
//     if (rank < size - 1)
//         MPI_Sendrecv(local_chunk.ptr(local_rows), cols, MPI_UNSIGNED_CHAR, rank + 1, 0,
//                      local_chunk.ptr(local_rows + 1), cols, MPI_UNSIGNED_CHAR, rank + 1, 0,
//                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);

//     // Apply filter
//     Mat filtered(local_rows, cols, CV_8UC1);
//     for (int i = 1; i <= local_rows ; ++i)
//         for (int j = 0; j < cols; ++j)
//             filtered.at<uchar>(i - 1, j) = apply_kernel(local_chunk, i, j, kernel, ksize);

//     // Gather result
//     Mat final_image;
//     if (rank == 0)
//         final_image = Mat(rows, cols, CV_8UC1);

//     MPI_Gatherv(filtered.data, local_rows * cols, MPI_UNSIGNED_CHAR,
//                 final_image.data, sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
//                 0, MPI_COMM_WORLD);

//     if (rank == 0) {
//         auto end = chrono::high_resolution_clock::now();
//         chrono::duration<double> elapsed = end - start;
//         cout << "Time taken: " << elapsed.count() << " seconds" << endl;
//         imwrite("output_3.jpg", final_image);
//         cout << "High-pass filter applied and saved to output.jpg" << endl;
//         // 🖼️ Show original and filtered images
//         imshow("Original Image", image);
//         imshow("Filtered Image", final_image);
//         waitKey(0);  // Wait for key press
//         destroyAllWindows();  // Close all OpenCV windows
//     }

//     MPI_Finalize();
//     return 0;
// }

#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>


using namespace std;
using namespace cv;

//CV_8UC1 --> 8 bit unsigned integer and c1 is one channel ( as it is greyscale)

std::chrono::time_point<std::chrono::high_resolution_clock> start;




vector<vector<int>> generateKernel(int ksize) {
    vector<vector<int>> kernel(ksize, vector<int>(ksize, -1));
    kernel[ksize / 2][ksize / 2] = ksize * ksize - 1;
    return kernel;
}

uchar apply_kernel(const Mat& img, int row, int col, const vector<vector<int>>& kernel, int ksize) {
    int sum = 0;
    int offset = ksize / 2;
    for (int i = -offset; i <= offset; ++i) {
        for (int j = -offset; j <= offset; ++j) {
            int r = min(max(row + i, 0), img.rows - 1);
            int c = min(max(col + j, 0), img.cols - 1);
            sum += img.at<uchar>(r, c) * kernel[i + offset][j + offset];
        }
    }
    return saturate_cast<uchar>(sum);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0)
            cerr << "Usage: mpirun -np <num_processes> ./MPI_filter <image_path> <kernel_size>" << endl;
        MPI_Finalize();
        return 1;
    }

    string image_path = argv[1];
    int ksize = atoi(argv[2]);
    vector<vector<int>> kernel = generateKernel(ksize);

    if (ksize < 3 || ksize % 2 == 0) {
        if (rank == 0)
            cerr << "Kernel size must be odd and >= 3." << endl;
        MPI_Finalize();
        return 1;
    }

    Mat image, gray;
    int rows, cols;

    if (rank == 0) {
        image = imread(image_path, IMREAD_GRAYSCALE);
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
        vector<uchar> flat_image(image.begin<uchar>(), image.end<uchar>());
        MPI_Scatterv(flat_image.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
                     local_chunk.ptr(1), local_rows * cols, MPI_UNSIGNED_CHAR,
                     0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_UNSIGNED_CHAR,
                     local_chunk.ptr(1), local_rows * cols, MPI_UNSIGNED_CHAR,
                     0, MPI_COMM_WORLD);
    }

    if (rank > 0)
        MPI_Sendrecv(local_chunk.ptr(1), cols, MPI_UNSIGNED_CHAR, rank - 1, 0,
                     local_chunk.ptr(0), cols, MPI_UNSIGNED_CHAR, rank - 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (rank < size - 1)
        MPI_Sendrecv(local_chunk.ptr(local_rows), cols, MPI_UNSIGNED_CHAR, rank + 1, 0,
                     local_chunk.ptr(local_rows + 1), cols, MPI_UNSIGNED_CHAR, rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    Mat filtered(local_rows, cols, CV_8UC1);
    for (int i = 1; i <= local_rows; ++i)
        for (int j = 0; j < cols; ++j)
            filtered.at<uchar>(i - 1, j) = apply_kernel(local_chunk, i, j, kernel, ksize);

    Mat final_image;
    if (rank == 0)
        final_image = Mat(rows, cols, CV_8UC1);

    MPI_Gatherv(filtered.data, local_rows * cols, MPI_UNSIGNED_CHAR,
                final_image.data, sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    // if (rank == 0) {
    //     auto end = chrono::high_resolution_clock::now();
    //     chrono::duration<double> elapsed = end - start;

    //     string output_path = "Output/mpi/kernel_" + to_string(ksize) + ".jpg";
    //     imwrite(output_path, final_image);
    //     cout << "High-pass filter applied and saved to " << output_path << endl;
    //     cout << "Time taken: " << elapsed.count() << " seconds" << endl;

    //     // Optional live preview (manual runs)
    //     imshow("Filtered Image", final_image);
    //     waitKey(0);
    //     destroyAllWindows();
    // }
    if (rank == 0) {
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
    
        // Save the filtered image
        string outputFileName = "Output/mpi/kernel_" + to_string(ksize) + ".jpg";
        imwrite(outputFileName, final_image);
        cout << "Filtered image with kernel size " << ksize << " saved successfully as " << outputFileName << ".\n";
    
        // // Display the filtered image
        // imshow("Filtered Image", final_image);
        // waitKey(0);
        // destroyAllWindows();
    
        // Print the elapsed time
        cout << "Execution time for kernel size " << ksize << ": " << elapsed.count() << " seconds.\n";
    }
    

    MPI_Finalize();
    return 0;
}
