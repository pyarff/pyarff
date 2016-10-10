# distutils: language = c++

# Authors: Raghav R V <rvraghav93@gmail.com>
#
# Licence: BSD 3 clause

# CPP libraries
from libcpp.string cimport string
from libcpp.map cimport map as cpp_hashmap
from libcpp.vector cimport vector

# C libraries
from libc.stdio cimport FILE, fopen, fclose
from libc.stdio cimport fgetc, getline, feof
from libc.stdlib cimport free, malloc

# Cython imports
cimport numpy as cnp
from .utils cimport string_to_double

# Python imports
import os
import re

from scipy.sparse import issparse
from scipy.sparse import csr_matrix

import numpy as np

# Consts to define various @attribute types
cdef UINT_8 ATTR_NUMERIC = 0  # np.int
cdef UINT_8 ATTR_INTEGER = 1
cdef UINT_8 ATTR_REAL = 2
cdef UINT_8 ATTR_NOMINAL = 3
cdef UINT_8 ATTR_NOMINAL_ENCODED = 4

# We don't handle the following
cdef UINT_8 ATTR_STRING = 5
cdef UINT_8 ATTR_DATE = 6
cdef UINT_8 ATTR_RELATIONAL = 7

ATTR_TYPE_INDEX_TO_NP_TYPE_ENC_NOMINAL = {
    ATTR_NUMERIC: np.int32,
    ATTR_INTEGER: np.int32,
    ATTR_REAL: np.float64,
    ATTR_NOMINAL_ENCODED: np.int32,
    ATTR_NOMINAL: np.unicode_,
    ATTR_STRING: np.unicode_,
    ATTR_DATE: np.unicode_,  # Will be considered as a string data
}

# The maximum size for a single attribute field
cdef size_t MAX_ATTRIBUTE_LEN = 10000

cdef class ARFFReader(object):
    """ARFFReader is used to read from arff formatted data files."""

    def __cinit__(self, string filename, bint encode_nominals):
        # Store the filename as an attribute so we don't have to pass it around.
        self.filename = filename
        # Whether or not to encode the nominal values to a string index
        self.encode_nominals = encode_nominals

        # Data dependent parameters
        self.sparse = False
        self.missing_data = False

        # Initialize new vectors (Lets store these in the heap)
        self.attribute_names = new vector[string]()
        self.attribute_types = new vector[UINT_8]()
        self.attribute_types_str = new vector[string]()
        self.categories = new vector[cpp_hashmap[string, size_t]]()
        self.categorical = new vector[bint]()
        self.n_categories = new vector[size_t]()
        self.convertors = new vector[UINT_8]()

        self.n_samples = 0
        self.n_attributes = 0
        self.data_header_line_no = 0

        self.relation = ""

    def __dealloc__(self):
        del self.attribute_names
        del self.attribute_types
        del self.attribute_types_str
        del self.categories
        del self.categorical
        del self.n_categories
        del self.convertors

    cdef void _load_metadata(self):
        """Read the attributes from the arff datafile and update class metadata"""
        cdef size_t line_no
        cdef size_t i
        cdef UINT_8 attr_type
        cdef size_t n_attributes = 0
        cdef bint metadata_read = False
        cdef bint is_categorical = False

        # The categories hashmap pointer for the current attribute
        cdef cpp_hashmap[string, size_t] *categories_i = NULL

        # Clear the previous vector entries??
        # self.clear()

        # REGEX to extract the attr name and value
        # To get attributes name when there are no quotes
        r_unquoted_attr_name = re.compile(r"(\S+)\s+(..+$)")

        # To get attributes name enclosed with single quotes - ''
        r_quoted_attr_name = re.compile(r"'(..+)'\s+(..+$)")

        # Since the header information is very small, we'll use the simpler
        # python syntax to process it.

        with open(self.filename, 'r') as arff_file:
            for line_no, raw_line in enumerate(arff_file):

                line = raw_line.strip()

                # Skip comment lines
                if line[0] == '%':
                    continue

                first_field = line[:10].lower()

                # Check for @attribute or @relation
                if first_field == '@attribute':
                    # NOTE ARFF format does not define multiline attributes or
                    # data
                    attr_value_str = line[10:].strip()
                    category_list = line[10:].strip()
                    is_categorical = False

                    regex_match = re.match(r_quoted_attr_name, attr_value_str)
                    if not regex_match:
                        # Check for quoted attribute names.
                        regex_match = re.match(r_unquoted_attr_name, attr_value_str)

                    if not regex_match:
                        raise ValueError("Incorrect @attribute format : %s" %
                                         attr_value_str)

                    attr_name = bytes(regex_match.group(1), 'utf-8')

                    # Understand the type of attribute (The 2nd group in attr_name regex)
                    attr_value_str = regex_match.group(2)
                    attr_type_str = attr_value_str[:12].lower()

                    if attr_value_str[0] == "{":
                        if self.encode_nominals:
                            attr_type = ATTR_NOMINAL_ENCODED
                            attr_type_str = b'nominal (integer encoded)'
                        else:
                            attr_type = ATTR_NOMINAL
                            attr_type_str = b'nominal'


                        category_list = attr_value_str.strip("{ }").split(',')

                        # Clean up and remove quotes
                        # "'"/'"' --> b"'"/b'"'
                        # ' --> b"'"
                        # 'abc def' --> b"abc def"
                        # 'abc' --> b"abc"
                        # TODO "" --> ERROR?

                        category_list = list(map(_clean_cat_names,
                                                 category_list))
                        # print(category_list)
                        is_categorical = True

                    elif attr_type_str.startswith('real'):
                        attr_type = ATTR_REAL
                        attr_type_str = b'real'

                    elif attr_type_str.startswith('integer'):
                        attr_type = ATTR_INTEGER
                        attr_type_str = b'integer'

                    elif attr_type_str.startswith('numeric'):
                        attr_type = ATTR_NUMERIC
                        attr_type_str = b'numeric'

                    elif attr_type_str.startswith('string'):
                        attr_type = ATTR_STRING
                        attr_type_str = b'string'
                        raise NotImplementedError("STRING type attributes are not supported yet.")

                    elif attr_type_str.startswith('date'):
                        attr_type = ATTR_DATE
                        attr_type_str = b'date'
                        raise NotImplementedError("DATE type attributes are not supported yet.")

                    elif attr_type_str.startswith('relational'):
                        attr_type = ATTR_RELATIONAL
                        attr_type_str = b'relational'
                        raise NotImplementedError("RELATIONAL type attributes are not supported yet.")

                    else:
                        raise ValueError("Unknown attribute - %s" % attr_type_str)

                    # Record the type and value (if any)
                    n_attributes += 1

                    categories_i = new cpp_hashmap[string, size_t]()
                    # Index categories from 1, as default key-not-found return value is 0
                    for i, category in enumerate(category_list, start=1):
                        # 0 indexing to dereference the pointer
                        (categories_i[0])[bytes(category, 'utf-8')] = i

                    # TODO Add a missing category?

                    # 0 indexing to dereference the pointer

                    # Store all the categories for this attribute
                    (self.categories[0]).push_back(categories_i[0])
                    (self.n_categories[0]).push_back(i)
                    (self.categorical[0]).push_back(is_categorical)

                    # Store the type for this attribute
                    (self.attribute_types[0]).push_back(attr_type)
                    # Store the string type field, so we can get a dict of the dataset meta info
                    (self.attribute_types_str[0]).push_back(attr_type_str)

                    # Store the name for this attribute
                    (self.attribute_names[0]).push_back(attr_name)

                elif first_field.startswith('@relation'):
                    self.relation = bytes(line[9:].strip(), 'utf-8')

                elif first_field.startswith('@data'):
                    metadata_read = True
                    self.n_attributes = n_attributes
                    # The data_line_no is the 0-indexed number of the line at which
                    # the "@DATA" meta header is found
                    self.data_header_line_no = line_no
                    # print "Data at", line_no
                    break

    cdef void _scan_data(self):
        cdef size_t n_samples = 0
        cdef data_header_line_no = self.data_header_line_no
        cdef bint sparse = self.sparse

        cdef char ch     # The current character
        cdef char first_char    # The first char of the parsed line
        cdef char* line = NULL  # The character buffer for the current line
        cdef size_t line_buffer_size = 0
        cdef size_t line_len = 0
        cdef size_t line_no = 0


        cdef FILE *cfile = fopen(self.filename.c_str(), 'r')
        if (cfile == NULL):
            raise ValueError("Cannot open file - %s" % self.filename)

        line_len = getline(&line, &line_buffer_size, cfile)

        # Read until the @DATA line
        while (line_len != -1) and (line_no < data_header_line_no):
            line_len = getline(&line, &line_buffer_size, cfile)
            line_no += 1

        # Read and parse the data
        line_len = 0
        while line_len != -1:
            line_len = getline(&line, &line_buffer_size, cfile)

            if line_len == -1:
                break

            first_char = line[0]

            # Or if the first char is '\r' (13), or '\n' (10), or '\0' (0)
            # Or if it is '%' (37), or ' ' <space> (32)
            if (first_char == 13 or first_char == 10 or first_char == 0 or
                    first_char == 37 or first_char == 32):
                continue

            # If the first char is '{' the data is sparse
            if not sparse and first_char == 123:
                sparse = True

            # XXX Does this slow down the scanning of data??
            if not missing_data and ('?' in line):
                missing_data = True

            n_samples += 1

        self.sparse = sparse
        self.n_samples = n_samples
        self.missing_data = missing_data

        free(line)    # Deallocate the line buffer
        fclose(cfile) # Close the file stream

    cdef void _read_data_dense(self, DOUBLE_t[:, :] data):
        cdef size_t n_samples = self.n_samples
        cdef size_t n_attributes = self.n_attributes

        cdef data_header_line_no = self.data_header_line_no
        cdef bint sparse = self.sparse

        cdef char ch     # The current character
        cdef char first_char    # The first char of the parsed line
        cdef char* line = NULL  # The character buffer for the current line
        cdef size_t line_buffer_size = 0
        cdef size_t char_index
        cdef size_t line_len = 0
        cdef size_t line_no = 0

        cdef size_t row
        cdef size_t col
        cdef bint inside_quotes
        cdef bint inside_attr_field
        cdef bint unparsed_attr
        cdef UINT_8 attr_type

        # To store the current attribute before conversion
        cdef char* attr_buffer = <char*>malloc(MAX_ATTRIBUTE_LEN * sizeof(char))
        cdef size_t attr_len = 0

        cdef FILE *cfile = fopen(self.filename.c_str(), 'r')
        if (cfile == NULL):
            raise ValueError("Cannot open file - %s" % self.filename)

        line_len = getline(&line, &line_buffer_size, cfile)

        # Read until the @DATA line (inclusive)
        while (line_len != -1) and (line_no < data_header_line_no):
            line_len = getline(&line, &line_buffer_size, cfile)
            line_no += 1

        # Read and parse the data
        line_len = 0
        row = 0
        while line_len != -1:
            line_len = getline(&line, &line_buffer_size, cfile)

            # EOF
            if line_len == -1:
                break

            first_char = line[0]

            # Empty line - if the first char is '\r' (13), or '\n' (10), or '\0' (0)
            # Comment - if it is '%' (37), or Invalid line - ' ' <space> (32)
            if (first_char == 13 or first_char == 10 or first_char == 0 or
                    first_char == 37 or first_char == 32):
                continue

            # print "Parsing data line", line,

            # Clear attr buffer variables
            attr_len = 0
            inside_quotes = False
            inside_attr_field = False

            unparsed_attr = False
            col = 0   # Attr number
            char_index = 0

            # dbg_str = "(%s) " % self.attribute_names[0][col]

            while char_index < line_len:
                # Process non-attribute characters - ,  '  \n  \r \0
                # To speed up, order the checks based on their frequency

                ch = line[char_index]

                # Skip all spaces that are not enclosed within quotes
                if ch == 32:
                    # dbg_str += "0 "
                    # Do not skip spaces inside quotes
                    if not inside_quotes:
                        # dbg_str += "1 "
                        while line[char_index] == 32:
                            char_index += 1
                        continue

                elif ch == 39:  # --> ' (quote)
                    if not inside_quotes:
                        # dbg_str +=  "2 "
                        inside_quotes = True
                        inside_attr_field = True
                        attr_buffer[attr_len] = '\0'
                        unparsed_attr = True
                        char_index += 1
                        # Don't continue go to end and parse the attr

                    # If inside quotes
                    else:
                        # dbg_str +=  "3 "
                        inside_quotes = False
                        # This quote terminates the attribute field
                        # Now lets skip all chars until the next comma
                        inside_attr_field = False
                        attr_buffer[attr_len] = '\0'
                        # Skip the quote character
                        char_index += 1
                        continue

                # comma in/out quotes
                elif ch == 44: # --> , (comma)
                    # dbg_str +=  "4 "
                    # If its a comma and we have already done processing the current attr
                    if not inside_attr_field:
                        # dbg_str +=  "5 "
                        if not unparsed_attr:
                            # dbg_str +=  "6 "
                            # Increment to next char and next attr
                            char_index += 1
                            continue

                    # If we aren't done processing the current attribute
                    else:
                        # dbg_str +=  "7 "
                        # Note that we've done storing the current attribute
                        # and that we need to process it.
                        inside_attr_field = False
                        attr_buffer[attr_len] = '\0'
                        char_index += 1
                        # Don't continue... go to the end and process the attr_buffer


                # Have to check both '\n' and '\r' for OS portability.
                elif ch == 10 or ch == 13 or ch == 0:
                    # dbg_str += "9 "
                    # Break if EOL '\n' or '\r' or '\0'
                    if not (inside_attr_field or unparsed_attr):
                        # dbg_str += "10 "
                        break

                    # If we reached the EOL and there is an unparsed attr (last col)
                    # Don't break
                    else:
                        # dbg_str += "11 "
                        # Note that we've done storing the current attribute
                        # and that we need to process it.
                        inside_attr_field = False

                # It reaches here if -
                #   it is a space inside a quote
                #   non-space non-meta chars inside/outside quotes
                # We need to store this in the attribute buffer
                else:
                    # dbg_str +=  "8 "
                    attr_buffer[attr_len] = ch
                    attr_len += 1
                    char_index += 1

                    # If unparsed_attr is set to False, inside_attr_field is
                    # also False
                    # But we can have a parsed attribute and incomplete attribute field
                    # (spaces inside the attribute field)
                    if not unparsed_attr:
                        # dbg_str +=  "9 "
                        unparsed_attr = True
                        inside_attr_field = True
                    continue

                # Now if we have unparsed attribute and we are not inside attribute field
                if not inside_attr_field and unparsed_attr:
                    # dbg_str +=  "- "
                    # parse and store the attribute to the data matrix
                    # Based on the type of the current attribute column

                    attr_type = (self.attribute_types[0])[col]
                    #print(col, ":", self.attribute_names[0][col],
                    #      attr_type, attr_buffer)

                    # If missing value store nan, no matter what the type is
                    if attr_buffer[0] == 63:
                        # print "missing"
                        data[row, col] = np.nan

                    elif (attr_type == ATTR_NUMERIC or
                              attr_type == ATTR_INTEGER or
                              attr_type == ATTR_REAL):
                        # print "converting to float"
                        data[row, col] = (
                            <DOUBLE_t> string_to_double(attr_buffer, attr_len))

                    elif attr_type == ATTR_NOMINAL_ENCODED:
                        # End the attr string with a NULL terminator and get
                        # the integer code

                        # print "converting to category index"
                        # if self.attribute_names[0][col] == b'class':
                        #     print(self.attribute_names[0][col])
                        #     print("\n\nSTART DEBUG")
                        #     print(<string>attr_buffer)
                        #     print(self.categories[0][col])
                        #     print(ch)
                        #     print(attr_len)
                        #     print("\n\nEND DEBUG")
                        attr_buffer[attr_len] = 0
                        data[row, col] = (
                            (self.categories[0])[col][<string> attr_buffer])
                        # Encoded category is 0 for unknown categories
                        if data[row, col] == 0.0:
                            raise ValueError("Unknown category (%s) detected "
                                             "for value of attribute %s for "
                                             "sample number %d"
                                             % (self.attribute_names[0][col],
                                                <string> attr_buffer, row))

                    col += 1
                    # Clear the attr variable
                    # dbg_str += "(%s) " % self.attribute_names[0][col]
                    attr_len = 0
                    unparsed_attr = False

            # Parse the next line
            # print(dbg_str)
            row += 1
            dbg_str = ""

        free(line)    # Deallocate memory from the line buffer
        free(attr_buffer)  # Deallocate memory from the attr_buffer
        fclose(cfile) # Close the file stream


    def get_metadata(self):
        """Load the dataset meta-data from the specified ARFF dataset.

        Returns
        -------

        meta_data_dict : dict
            The meta-data dict with the following keys - 
                relation : The arff dataset name (@RELATION field)
                filename : The given filename
                n_attributes : The total number of attributes
                n_samples : The total number of sample
        """
        if not os.path.exists(self.filename):
            raise ValueError("The dataset does not exist. The given filepath was %s" % self.filename)

        # Populate self with metadata information
        self._load_metadata()
        # Gets the n_samples, sparse, missing_data
        self._scan_data()

        meta_info = dict()
        meta_info['relation'] = self.relation
        meta_info['filename'] = self.filename
        meta_info['n_attrbutes'] = self.n_attributes

        all_attrs = dict()

        attribute_names_in_order = []

        for i in range(self.n_attributes):
            attr_name = (self.attribute_names[0])[i]
            attribute_names_in_order.append(attr_name)

            this_attr = dict()
            this_attr['order'] = i

            this_attr['type'] = (self.attribute_types_str[0])[i]
            attr_type_id = self.attribute_types[0][i]
            # this_attr['type_id'] = attr_type_id
            this_attr['numpy_type'] = (
                ATTR_TYPE_INDEX_TO_NP_TYPE_ENC_NOMINAL[attr_type_id])

            this_attr['is_categorical'] = (self.categorical[0])[i]
            if this_attr['is_categorical']:
                this_attr['n_categories'] = (self.n_categories[0])[i]
                this_attr['category_index_map'] = (
                    dict([cat for cat in ((self.categories[0])[i])]))

            all_attrs[attr_name] = this_attr

        meta_info['attributes'] = all_attrs
        meta_info['attribute_names_in_order'] = attribute_names_in_order
        meta_info['sparse'] = self.sparse
        meta_info['missing_data'] = self.missing_data
        meta_info['encode_nominals'] = self.encode_nominals

        meta_info['data_header_line_no'] = self.data_header_line_no
        meta_info['n_samples'] = self.n_samples

        return meta_info

    def get_data(self):
        data_shape = (self.n_samples, self.n_attributes)

        if self.sparse:
            data = csr_matrix(data_shape, dtype=np.float64)
            self._read_data_sparse(data)

        else:
            data = np.empty(data_shape, dtype=np.float64)
            self._read_data_dense(data)

        return data


def _clean_cat_names(catg):
    """Helper to clean categorical names and strip them of quotes."""
    catg = catg.strip()
    l_catg = len(catg)
    for quote_type in ("'", '"'):
        if catg[0] == quote_type:
            if l_catg == 1:
                # If the quote_type itself is a
                # category
                cleaned_catg = catg
            elif catg[-1] == quote_type:
                # Or if it simply surrounds the
                # category, strip it.
                cleaned_catg = catg.strip(quote_type)

    if len(cleaned_catg) == 0:
        raise ValueError("Bad category name - %s"
                         % catg)
    return cleaned_catg
