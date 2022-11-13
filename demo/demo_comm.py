
import llama
import numpy as np
np.seterr(divide='ignore')

from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()

d1 = np.random.uniform(0, 1, 100)
d2 = np.random.uniform(0, 1, 200)

axis = llama.AxisFactory.Regular("label", 4, 0, 1)
h1 = llama.Histogram(xaxis=axis)
h2 = llama.Histogram(xaxis=axis)

h1.fill(d1)
h2.fill(d2)

h3_add = h1 + h2
h3_mult = h1 * h2
h3_div = h1 / h2

h3_div_bcast = h3_div.bcast()

print(rank, "h1 =", h1.values())
print(rank, "h2 =", h2.values())
print(rank, "h1 + h2 =", h3_add.values())
print(rank, "h1 * h2 =", h3_mult.values())
print(rank, "h1 / h2 =", h3_div.values())
print(rank, "h3_div_bcast", h3_div_bcast.values())

