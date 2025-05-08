#include <iostream>
#include <opencv2/opencv.hpp>
#include "highpass.hpp"

using namespace cv;
using namespace std;

int main() {
    string imagePath;
    int kernelSize;
    int paddingChoice;

    cout << "Start\n"; 

    cout << "Enter path to input image: ";
    getline(cin, imagePath);
    cout << "Got image path: " << imagePath << endl;

    cout << "Enter kernel size (odd number like 3, 5, 7): ";
    cin >> kernelSize;

    cout << "Select padding method:\n";
    cout << "1. ZERO\n2. MIRROR\n3. REPLICATE\nYour choice: ";
    cin >> paddingChoice;

    PaddingType pad;
    switch (paddingChoice) {
        case 1: pad = ZERO; 
            break;
        case 2: pad = MIRROR; 
            break;
        case 3: pad = REPLICATE; 
            break;
        default:
            cerr << "Invalid choice. Defaulting to MIRROR.\n";
            pad = MIRROR;
    }

    Mat input = imread(imagePath, IMREAD_COLOR);
    if (input.empty()) {
        cerr << "Failed to load image: " << imagePath << endl;
        return -1;
    }

    // ---------- Your Custom High-Pass Filter ----------
    Mat customOutput;
    applyHighPassFilter(input, customOutput, kernelSize, pad);
    imwrite("output/sequential_output.jpg", customOutput);
    cout << "Saved custom output as output/sequential_output.jpg" << endl;

    // ---------- OpenCV Built-in High-Pass Filter ----------
    Mat builtInOutput;
    Mat kernel = (Mat_<float>(3, 3) <<
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1);

    filter2D(input, builtInOutput, -1, kernel);
    imwrite("output/opencv_output.jpg", builtInOutput);
    cout << "Saved OpenCV output as output/opencv_output.jpg" << endl;

    Mat output;
    applyHighPassFilter(input, output, kernelSize, pad);

    imwrite("output/sequential_output.jpg", output);
    cout << "Filtered image saved to output/sequential_output.jpg" << endl;

    return 0;
}
