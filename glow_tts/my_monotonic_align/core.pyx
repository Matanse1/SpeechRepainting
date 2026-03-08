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
        for x in range(max(0, t_x - 2*(t_y - y)+1), min(t_x, 2*y+1)):
            # Initialize default values
            v_cur = max_neg_val  # Stay move: (x, y-1)
            v_diag = max_neg_val  # Diagonal move: (x-1, y-1)
            v_jump = max_neg_val  # New move: (x-2, y-1)

            if x==0 and y==0:
                v_cur = 0
            else:
                if y > 0:
                    v_cur = value[x, y - 1]  # Stay
                if x > 0 and y > 0:
                    v_diag = value[x - 1, y - 1]  # Diagonal
                if x > 1 and y > 0:
                    v_jump = value[x - 2, y - 1]  # Jump

            # Update the current cell with the max score from all valid moves
            value[x, y] = max(v_cur, v_diag, v_jump) + value[x, y]

    for y in range(t_y - 1, -1, -1):
        path[index, y] = 1  # Mark the current position as part of the path.

        # Default move values
        move_jump = max_neg_val  # Invalid unless x > 1 and y > 0
        move_diagonal = max_neg_val  # Invalid unless x > 0 and y > 0
        move_stay = max_neg_val  # Invalid unless y > 0

        # Check bounds for the new move (x-2, y-1)
        if index > 1 and y > 0:
            move_jump = value[index-2, y-1]

        # Check bounds for diagonal move (x-1, y-1)
        if index > 0 and y > 0:
            move_diagonal = value[index-1, y-1]

        # Check bounds for staying (x, y-1)
        if y > 0:
            move_stay = value[index, y-1]

        # Debugging output to ensure indices are within bounds
        #with gil:
        #    print(f"At y={y}, index={index}, moves: jump={move_jump}, diag={move_diagonal}, stay={move_stay}")

        # Choose the move with the highest score
        if move_jump >= max(move_diagonal, move_stay):  # Prefer the new jump
            if index >= 2:  # Ensure index remains valid
                index = index - 2  # Move left two steps
        elif move_diagonal >= move_stay:  # Prefer diagonal
            if index >= 1:  # Ensure index remains valid
                index = index - 1  # Move left
        # Otherwise, stay in the same row (no change to `index`).


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void maximum_path_c(int8_t[:,:,::1] paths, float[:,:,::1] values, int16_t[::1] t_xs, int16_t[::1] t_ys, float max_neg_val=-1e9) nogil:
    cdef int b = values.shape[0]

    cdef int i
    for i in prange(b, nogil=True):
        maximum_path_each(paths[i], values[i], t_xs[i], t_ys[i], max_neg_val)
