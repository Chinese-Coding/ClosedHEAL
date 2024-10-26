from opencood.utils.cython.all_use cimport F32_t, F64_t, cnp
cpdef cnp.ndarray[F64_t, ndim=2] X1ToX2(cnp.ndarray[F64_t, ndim=1] x1, cnp.ndarray[F64_t, ndim=1] x2)