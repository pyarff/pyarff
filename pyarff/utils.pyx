cimport numpy as cnp


ctypedef cnp.npy_uint8 UINT_8            # 1 Byte wide integer datatype
ctypedef cnp.npy_intp SIZE_t             # Type for indices and counters
ctypedef cnp.npy_float64 DTYPE_t         # dtypes for the data matrix
                                         # Lets use 64 bit format as default float type

cpdef inline DTYPE_t string_to_double(char *p, SIZE_t total_len) nogil:
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


