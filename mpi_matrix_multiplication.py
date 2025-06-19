from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 400  # Matrix size
# --------------------------------------------------------- #

# The flow is like:
# comm.Bcast(B, root=0)                     # Share B to all processes
# comm.Scatter(A, A_chunk, root=0)          # Divide A rows
# C_chunk = np.matmul(A_chunk, B)           # Multiply
# comm.Gather(C_chunk, C, root=0)           # Gather C on rank 0

# --------------------------------------------------------- #

# Rank 0 generates full A and B
if rank == 0:
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    start_time = time.time()
else:
    A = None
    B = np.empty((N, N), dtype='d')

# Broadcast B to all processes
comm.Bcast(B, root=0)


if N % size != 0:
    if rank == 0:
        print(f"Matrix size {N} not divisible by number of processes {size}")
    MPI.Finalize()
    exit()

# Scatter rows of A
rows_per_proc = N // size
A_chunk = np.empty((rows_per_proc, N), dtype='d')
comm.Scatter(A, A_chunk, root=0)

# Each process computes its chunk of C
C_chunk = np.matmul(A_chunk, B)

# Gather the result chunks into full C at rank 0
if rank == 0:
    C = np.empty((N, N), dtype='d')
else:
    C = None

comm.Gather(C_chunk, C, root=0)

# Rank 0 prints time
if rank == 0:
    end_time = time.time()
    print(f"MPI matrix multiplication completed in {end_time - start_time:.4f} seconds")
