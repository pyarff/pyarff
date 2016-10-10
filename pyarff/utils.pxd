# distutils: language = c++

# Authors: Raghav R V <rvraghav93@gmail.com>
#
# Licence: BSD 3 clause

# See _pyarff.pyx for implementation details.

cimport numpy as cnp

# CPP libraries

ctypedef cnp.npy_float64 DOUBLE_t        # Type of y, sample_weight
ctypedef cnp.npy_uint8 UINT_8            # 1 Byte wide integer datatype
ctypedef cnp.npy_intp SIZE_t             # Type for indices and counters
ctypedef cnp.npy_float64 DTYPE_t         # dtypes for the data matrix
                                         # Lets use 64 bit format as default float type

# To convert string to double without overheads
cpdef double string_to_double(char *p, SIZE_t total_len) nogil
