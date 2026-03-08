import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.stdint cimport int8_t
from libc.stdint cimport int16_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void maximum_path_each(int8_t[:,::1] path, float[:,::1] value, int16_t t_x, int16_t t_y, float max_neg_val) nogil:
  cdef int x
  cdef int y
  cdef float v_prev
  cdef float v_cur
  cdef float tmp
  cdef int index = t_x - 1

  for y in range(t_y):
    for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
      if x == y:
        v_cur = max_neg_val
      else:
        v_cur = value[x, y-1]
      if x == 0:
        if y == 0:
          v_prev = 0.
        else:
          v_prev = max_neg_val
      else:
        v_prev = value[x-1, y-1]
      value[x, y] = max(v_cur, v_prev) + value[x, y]

  for y in range(t_y - 1, -1, -1):
    path[index, y] = 1
    if index != 0 and (index == y or value[index, y-1] < value[index-1, y-1]):
      index = index - 1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void maximum_path_c(int8_t[:,:,::1] paths, float[:,:,::1] values, int16_t[::1] t_xs, int16_t[::1] t_ys, float max_neg_val=-1e9) nogil:
  cdef int b = values.shape[0]

  cdef int i
  for i in prange(b, nogil=True):
    maximum_path_each(paths[i], values[i], t_xs[i], t_ys[i], max_neg_val)
