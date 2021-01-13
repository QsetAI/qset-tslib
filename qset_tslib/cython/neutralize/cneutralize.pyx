# distutils: language = c++
from __future__ import division

import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t

from libcpp cimport bool
from libcpp.map cimport map

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double] neutralize_row(np.ndarray[DTYPE_t, ndim=1] v, list isfinite, list group, bool norm_std=False):
    cdef int n_ids = 0
    cdef int n = v.shape[0]

    cdef map[int, int] ids
    
    for i in range(n):
        if isfinite[i] == 1:
            if ids.find(group[i]) == ids.end():
                ids[group[i]] = n_ids
                n_ids += 1

    cdef np.ndarray[DTYPE_t, ndim=1] s = np.zeros(n_ids, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] s2 = np.zeros(n_ids, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] cnt = np.zeros(n_ids, dtype=DTYPE)
    
    cdef int cur_id;

    # calculate sums and stds
    for i in range(n):
        if isfinite[i] == 1:
            cur_id = ids[group[i]]
            s[cur_id] += v[i]
            s2[cur_id] += v[i] ** 2
            cnt[cur_id] += 1

    cdef np.ndarray[DTYPE_t, ndim=1] res = np.full(n, np.nan)
    
    s /= cnt
    
    if not norm_std:
        for i in range(n):
            if isfinite[i] == 1:
                res[i] = v[i] - s[ids[group[i]]]
    else:
        s2 = np.sqrt((s2 - cnt * (s ** 2)) / (cnt - 1))
        for i in range(n):
            if isfinite[i] == 1:
                cur_id = ids[group[i]]
                res[i] = np.float64((v[i] - s[cur_id])) / s2[cur_id]
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double] neutralize(np.ndarray v, np.ndarray group, bool norm_std=False):
    cdef int T = v.shape[0]
    cdef int n = v.shape[1]

    assert T == group.shape[0]
    assert n == group.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=2] res = np.empty((T, n), dtype=DTYPE)
    cdef np.ndarray[np.uint8_t, ndim=2] isfinite = np.isfinite(v).astype(np.uint8)
    
    assert v.dtype == DTYPE
    
    for t in range(T):
        res[t, :] = neutralize_row(v[t, :], isfinite[t, :].tolist(), group[t, :].tolist(), norm_std=norm_std)
    return res