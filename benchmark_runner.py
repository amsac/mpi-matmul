import subprocess
import matplotlib.pyplot as plt
import re

process_counts = [2, 5 , 8]
times = []

for n in process_counts:
    print(f"Running with {n} processes...")
    try:
        # Run the MPI script
        result = subprocess.run(
            ["mpiexec", "-n", str(n), "python", "mpi_matrix_multiplication.py"],
            capture_output=True,
            text=True,
            check=True
        )
        # Extract the time from stdout
        match = re.search(r"completed in ([\d.]+) seconds", result.stdout)
        if match:
            exec_time = float(match.group(1))
            times.append(exec_time)
            print(f"Time: {exec_time:.4f} seconds")
        else:
            print("Time not found in output.")
            times.append(None)

    except subprocess.CalledProcessError as e:
        print(f"Error running with {n} processes:\n{e.stderr}")
        times.append(None)

# Plotting
speedup = [times[0] / t if t else 0 for t in times]

plt.figure(figsize=(8, 5))
plt.plot(process_counts, speedup, marker='o', color='blue')
plt.title("Speedup vs Number of Processes")
plt.xlabel("Number of Processes")
plt.ylabel("Speedup")
plt.grid(True)
plt.xticks(process_counts)
plt.tight_layout()
plt.show()
