import numpy as np

from io import StringIO
from scipy.io import mmread, mmwrite

matrix = "494_bus"

fin = f"../matrices/symmetric/{matrix}.mtx"
fout = f"../matrices/symmetric/{matrix}_rhs.mtx"

m = mmread(fin, spmatrix=True)
nrow, ncol = m.shape

print(nrow, ncol)

rhs = np.random.rand(1, ncol) * 100
mmwrite(fout, rhs)
