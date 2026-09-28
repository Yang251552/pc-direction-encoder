import os

# Another agent trains in parallel on this 4-core box: keep our BLAS/OpenMP pools small.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "2")
