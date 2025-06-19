import subprocess
import matplotlib.pyplot as plt
import re

process_counts = [1, 2, 4, 8]
times = []

def run_and_get_time(command, label):
    print(f"Running: {label}")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        match = re.search(r"completed in ([\d.]+) seconds", result.stdout)
        if match:
            exec_time = float(match.group(1))
            print(f"{label} time: {exec_time:.4f} seconds")
            return exec_time
        else:
            print("Time not found in output.")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error:\n{e.stderr}")
        return None

# Run serial implementation (1 process)
serial_time = run_and_get_time(
    ["python", "serial_matrix_multiplication.py"],
    "Serial"
)
times.append(serial_time)

# Run MPI implementation with 2, 4, 8 processes
for n in process_counts[1:]:
    time_n = run_and_get_time(
        ["mpiexec", "-n", str(n), "python", "mpi_matrix_multiplication.py"],
        f"MPI with {n} processes"
    )
    times.append(time_n)

# Plot execution time
plt.figure(figsize=(8, 5))
plt.plot(process_counts, times, marker='o', label="Execution Time", color='green')
plt.axhline(y=serial_time, color='red', linestyle='--', label="Serial Baseline")

plt.title("Execution Time vs Number of Processes")
plt.xlabel("Number of Processes")
plt.ylabel("Time (s)")
plt.xticks(process_counts)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: plot speedup
speedup = [serial_time / t if t else 0 for t in times]

plt.figure(figsize=(8, 5))
plt.plot(process_counts, speedup, marker='o', color='blue', label="Speedup")
plt.title("Speedup vs Number of Processes")
plt.xlabel("Number of Processes")
plt.ylabel("Speedup")
plt.xticks(process_counts)
plt.grid(True)
plt.tight_layout()
plt.show()

