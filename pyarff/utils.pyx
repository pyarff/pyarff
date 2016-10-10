# distutils: language = c++
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True


# Authors: Raghav R V <rvraghav93@gmail.com>
#
# Licence: BSD 3 clause

# See _pyarff.pyx for implementation details.


cimport numpy as cnp

cpdef DOUBLE_t string_to_double(char *p, SIZE_t total_len) nogil:
    """To convert string to double without overheads"""
    cdef DTYPE_t res = 0.0
    cdef bint neg = False
    cdef DTYPE_t d_place_value = 0.1

    cdef size_t i = 0

    if p[i] == 45:  # '-'
        neg = True
        i += 1

    # The integer part
    while i < total_len:
        if p[i] == 46:
            i += 1
            break

        res *= 10.
        res += (p[i] - 48)
        i += 1

    # The decimal part
    d_place_value = 0.1
    while i < total_len:
        res += (p[i] - 48) * d_place_value
        d_place_value *= 0.1
        i += 1

    if neg:
        return -res
    else:
        return res



cpdef SIZE_t string_to_int(char *p, SIZE_t total_len) nogil:
    """To convert string to int without overheads"""
    cdef size_t res = 0
    cdef bint neg = False

    cdef size_t i = 0

    if p[i] == 45:  # '-'
        neg = True
        i += 1

    # The integer part
    while i < total_len:
        res *= 10
        res += (p[i] - 48)
        i += 1

    if neg:
        return -res
    else:
        return res
