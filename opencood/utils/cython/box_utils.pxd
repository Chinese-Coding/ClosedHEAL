from opencood.utils.cython.all_use cimport F32_t, F64_t, cnp

cpdef cnp.ndarray[F64_t, ndim=3] BoxesToCorners3D(cnp.ndarray[F64_t, ndim=2] boxes3d, str order)

cpdef cnp.ndarray[F64_t, ndim=2] Corner2dToStandupBox(cnp.ndarray[F64_t, ndim=3] box2d)