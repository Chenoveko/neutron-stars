from numba import njit
import numpy as np
import time

x = np.arange(100).reshape(10, 10)


@njit(parallel=True,fastmath=True)
def go_fast_parallel(a):  # Function is compiled and runs in machine code
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace


@njit(fastmath=True)
def go_fast(a):  # Function is compiled and runs in machine code
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace


def go_slow(a):  # Function is compiled and runs in machine code
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace


# NORMAL PYTHON
start = time.perf_counter()
go_slow(x)
end = time.perf_counter()
print("Elapsed in normal python 1 = {}s".format((end - start)))

start = time.perf_counter()
go_slow(x)
end = time.perf_counter()
print("Elapsed in normal python 2 = {}s".format((end - start)))

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.perf_counter()
go_fast(x)
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.perf_counter()
go_fast(x)
end = time.perf_counter()
print("Elapsed (after compilation) = {}s".format((end - start)))

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.perf_counter()
go_fast_parallel(x)
end = time.perf_counter()
print("Elapsed parallel (with compilation) = {}s".format((end - start)))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.perf_counter()
go_fast_parallel(x)
end = time.perf_counter()
print("Elapsed parallel (after compilation) = {}s".format((end - start)))
