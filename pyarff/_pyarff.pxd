# distutils: language = c++
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

# Authors: Raghav R V <rvraghav93@gmail.com>
#
# Licence: BSD 3 clause

# See _pyarff.pyx for implementation details.

# CPP libraries
from libcpp.string cimport string
from libcpp.map cimport map as cpp_hashmap
from libcpp.vector cimport vector

# Python imports
cimport numpy as cnp

ctypedef cnp.npy_float64 DOUBLE_t         # Type of X, y, sample_weight
ctypedef cnp.npy_intp SIZE_t              # Type for indices and counters
ctypedef unsigned char UINT_8             # 1 Byte wide integer datatype

cdef class ARFFReader(object):
    cdef readonly bint encode_nominals   # Whether or not to encode the nominal
                                         #(categorical) values

    cdef readonly bint sparse     # Whether to use sparse or dense data matrix
    cdef readonly bint missing_data     # Whether to look for ? while parsing

    cdef size_t n_samples           # The total number of rows/samples (shape[0] of the data matrix)
    cdef size_t n_attributes              # The total number of cols/attributes (shape[1] of the data matrix)
    cdef size_t data_header_line_no       # The zero-indexed line number of the line which contains the @DATA header

    cdef vector[string] *attribute_names       # The attribute names
    cdef vector[UINT_8] *attribute_types       # The attribute types for internal use (ATTR_NUMERIC etc)
    cdef vector[string] *attribute_types_str   # The attribute types as string for external use ("numeric" etc)
    cdef vector[cpp_hashmap[string, size_t]] *categories
                                               # A vector of CPP hashmaps mapping the string category names to integer values

    cdef vector[bint] *categorical     # To denote if the attribute is nominal(categorical) or not
    cdef vector[size_t] *n_categories  # To denote the number of categories in each categorical attribute
    cdef vector[UINT_8] *convertors    # c-level arrys of 1 byte converter ids to denote the id of
                                       # converter(parser) to use to convert the string attribute to actual data.

    cdef readonly string filename      # The filename of the arff file.
    cdef readonly string relation      # The name of the arff dataset (@RELATION)


    # Methods
    cdef void _load_metadata(self) except+
    cdef void _scan_data(self) except+
    cdef void _read_data_dense(self, DOUBLE_t[:, :] data) except+
