import pandas as pd

# Load result files
file1 = "results_seq_omp.txt"  # From Sequential and OpenMP runs
file2 = "results_mpi.txt"      # From MPI runs

# Read both files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Combine both into one DataFrame
df = pd.concat([df1, df2], ignore_index=True)

# Optional: pivot to better view
pivot = df.pivot_table(index=["Image", "Kernel Size"],
                       columns="Method",
                       values="Time (s)",
                       aggfunc="first")

# Sort for clarity
pivot = pivot.sort_index()

# Save to CSV or print
pivot.to_csv("merged_comparison_table.csv")
print(pivot)
