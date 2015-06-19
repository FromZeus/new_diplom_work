import numpy as np
cimport numpy as cnp

def func(cnp.ndarray[cnp.int64_t, ndim=2] ext_mem = np.array([np.arange(5)] * 5)):
	print ext_mem
	arr = ext_mem
	if not arr:
		arr = np.array([np.zeros(5).astype("float64")] * 5)
	print arr