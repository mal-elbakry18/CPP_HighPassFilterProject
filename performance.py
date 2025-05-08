import os
import subprocess
import time
import cv2
from pathlib import Path

# Configuration
input_dir = "Input"
output_base = "Output_Performance"
results_file = "results.txt"

executables = {
    "Sequential": "./sequential_2",
    "OpenMP": "./openMP_filter",
    "MPI": "mpirun -np 4 ./MPI_filter_2"
}
kernel_sizes = [3, 5, 7, 9, 11]
ignore_files = {"download.jpeg", "lena.png"}

# Clear previous results
with open(results_file, "w") as f:
    f.write("Method,Image,Kernel Size,Time (s),Status,Output Path\n")

# Collect valid test images
image_files = []
for f in os.listdir(input_dir):
    if f.lower().endswith((".jpg", ".jpeg", ".png")) and Path(f).name not in ignore_files:
        image_files.append(os.path.join(input_dir, f))

# Sort by resolution to find the largest for stress test
def image_resolution(path):
    img = cv2.imread(path)
    return 0 if img is None else img.shape[0] * img.shape[1]

image_files.sort(key=image_resolution, reverse=True)
largest_image = image_files[0] if image_files else None

# Benchmark each image with each kernel
for ksize in kernel_sizes:
    print(f"\n=== Kernel Size: {ksize} ===")
    kernel_output_dir = os.path.join(output_base, f"kernel_{ksize}")
    os.makedirs(kernel_output_dir, exist_ok=True)

    for image_path in image_files:
        image_name = Path(image_path).stem
        for label, command in executables.items():
            out_name = f"{image_name}_{label.lower()}.jpg"
            output_path = os.path.join(kernel_output_dir, out_name)
            full_cmd = f"{command} {image_path} {ksize}"

            print(f"{label} | {image_name} | k={ksize}", end=": ")

            try:
                start = time.time()
                result = subprocess.run(full_cmd.split(), capture_output=True, text=True)
                elapsed = time.time() - start

                if result.returncode == 0:
                    original_folder = {
                        "Sequential": "Output/sequential",
                        "OpenMP": "Output/openMP",
                        "MPI": "Output/mpi"
                    }[label]
                    original_filename = f"kernel_{ksize}.jpg"
                    original_path = os.path.join(original_folder, original_filename)

                    if os.path.exists(original_path):
                        os.rename(original_path, output_path)
                        status = "Success"
                        print(f"{elapsed:.2f}s → {output_path}")
                    else:
                        status = "Missing Output"
                        print(f"{elapsed:.2f}s but image not found at {original_path}")
                else:
                    status = "Error"
                    print(f"Failed\n{result.stderr}")

                with open(results_file, "a") as f:
                    f.write(f"{label},{image_name},{ksize},{elapsed:.2f},{status},{output_path}\n")

            except Exception as e:
                print(f"Exception: {e}")
                with open(results_file, "a") as f:
                    f.write(f"{label},{image_name},{ksize},-,Exception,{output_path}\n")

# Stress test
if largest_image:
    ksize = 101
    print(f"\n=== Stress Test: Kernel 101 on {Path(largest_image).name} ===")
    kernel_output_dir = os.path.join(output_base, f"kernel_{ksize}")
    os.makedirs(kernel_output_dir, exist_ok=True)

    image_name = Path(largest_image).stem
    for label, command in executables.items():
        out_name = f"{image_name}_{label.lower()}.jpg"
        output_path = os.path.join(kernel_output_dir, out_name)
        full_cmd = f"{command} {largest_image} {ksize}"

        print(f"{label} | {image_name} | k={ksize}", end=": ")
        try:
            start = time.time()
            result = subprocess.run(full_cmd.split(), capture_output=True, text=True)
            elapsed = time.time() - start

            if result.returncode == 0:
                original_folder = {
                    "Sequential": "Output/sequential",
                    "OpenMP": "Output/openMP",
                    "MPI": "Output/mpi"
                }[label]
                original_filename = f"kernel_{ksize}.jpg"
                original_path = os.path.join(original_folder, original_filename)

                if os.path.exists(original_path):
                    os.rename(original_path, output_path)
                    status = "Success"
                    print(f"{elapsed:.2f}s → {output_path}")
                else:
                    status = "Missing Output"
                    print(f"{elapsed:.2f}s but image not found at {original_path}")
            else:
                status = "Error"
                print(f"Failed\n{result.stderr}")

            with open(results_file, "a") as f:
                f.write(f"{label},{image_name},{ksize},{elapsed:.2f},{status},{output_path}\n")

        except Exception as e:
            print(f"Exception: {e}")
            with open(results_file, "a") as f:
                f.write(f"{label},{image_name},{ksize},-,Exception,{output_path}\n")
