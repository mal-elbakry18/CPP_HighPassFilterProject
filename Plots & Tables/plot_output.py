import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# === Load and combine both result files ===
file1 = "results_seq_omp.txt"
file2 = "results_mpi.txt"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

df = pd.concat([df1, df2], ignore_index=True)
df = df[df["Status"] == "Success"]

# === Extract resolution from output images ===
def get_resolution(path):
    try:
        img = cv2.imread(path)
        if img is not None:
            return img.shape[0] * img.shape[1]
    except:
        return 0
    return 0

df["Resolution"] = df["Output Path"].apply(get_resolution)

# === Plot 1: Resolution vs Execution Time ===
plt.figure(figsize=(10, 6))
for method in df["Method"].unique():
    subset = df[df["Method"] == method]
    plt.scatter(subset["Resolution"], subset["Time (s)"], label=method)

plt.title("Impact of Image Resolution on Execution Time")
plt.xlabel("Image Resolution (Total Pixels)")
plt.ylabel("Execution Time (s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
Path("plots").mkdir(exist_ok=True)
plt.savefig("plots/resolution_vs_time.png")
plt.show()

# === Plot 2: Kernel Size vs Execution Time (Line Plot) ===
plt.figure(figsize=(10, 6))
for method in df["Method"].unique():
    subset = df[df["Method"] == method]
    avg_by_kernel = subset.groupby("Kernel Size")["Time (s)"].mean()
    plt.plot(avg_by_kernel.index, avg_by_kernel.values, marker='o', label=method)

plt.title("Impact of Kernel Size on Execution Time")
plt.xlabel("Kernel Size")
plt.ylabel("Average Execution Time (s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plots/kernel_vs_time.png")
plt.show()
