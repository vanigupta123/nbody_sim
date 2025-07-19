import numpy as np
import matplotlib.pyplot as plt
import faiss

d = 64 # dimension
nb = 100000 # db size
nq = 10000 # number of queries
np.random.seed(1234) # causes random number generator to start at the same initial (seed) state
xb = np.random.random((nb, d)).astype('float32') # matrix that's 100,000x64
xb[:, 0] += np.arange(nb) / 1000. # xb[:, 0] has dimension 100,000x1
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

index = faiss.IndexFlatL2(d)
print(index.is_trained)
index.add(xb) # add vectors to index -- IDs are assigned to each vector
# other index types can store custom IDs, instead of assigning numerical order IDs
print(index.ntotal)

k = 4
D, I = index.search(xq, k) # integer matrix of size nq x k
print(D) # nq x k matrix, where row i contains the IDs of query vector i, sorted by increasing distance
print(I) # nq x k matrix of squared distances
print(I[:5])
print(I[-5:])