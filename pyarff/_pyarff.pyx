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

ctypedef cnp.npy_uint8 UINT_8            # 1 Byte wide integer datatype
ctypedef cnp.npy_intp SIZE_t             # Type for indices and counters
ctypedef cnp.npy_float64 DTYPE_t         # dtypes for the data matrix
                                         # Lets use 64 bit format as default float type

# Consts to define various @attribute types
cdef UINT_8 ATTR_NUMERIC = 0
cdef UINT_8 ATTR_INTEGER = 1
cdef UINT_8 ATTR_REAL = 2
cdef UINT_8 ATTR_NOMINAL = 3

# We don't handle the following
cdef UINT_8 ATTR_STRING = 4
cdef UINT_8 ATTR_DATE = 5
cdef UINT_8 ATTR_RELATIONAL = 6

# The maximum size for a single attribute field
cdef size_t MAX_ATTRIBUTE_LEN = 10000

cdef class ARFFReader(object):
    """ARFFReader is used to read from arff formatted data files."""

    def __cinit__(self, str filename, bint encode_labels):
        # Store the filename as an attribute so we don't have to pass it around.
        self.filename = filename
        # Whether or not to encode the nominal values to a string index
        self.encode_labels = encode_labels
        
        # Data dependent parameters
        self.is_data_sparse = False
        self.missing_exist = False

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
        
    cdef _load_metadata(self):
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
            for line_no, line in enumerate(arff_file):
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

                    regex_match = re.match(r_unquoted_attr_name, attr_value_str)
                    if not regex_match:
                        # Check for quoted attribute names.
                        regex_match = re.match(r_quoted_attr_name, attr_value_str)

                    if not regex_match:
                        raise ValueError("Incorrect @attribute format : %s" %
                                         attr_value_str)

                    attr_name = regex_match.group(1)

                    # Understand the type of attribute (The 2nd group in attr_name regex)
                    attr_value_str = regex_match.group(2)
                    attr_type_str = attr_value_str[:12].lower()

                    if attr_value_str[0] == "{":
                        attr_type = ATTR_NOMINAL
                        attr_type_str = 'nominal'
                        
                        category_list = map(str.strip, attr_value_str.strip("{ }").split(','))
                        is_categorical = True                        

                    elif attr_type_str.startswith('real'):
                        attr_type = ATTR_REAL
                        attr_type_str = 'real'

                    elif attr_type_str.startswith('integer'):
                        attr_type = ATTR_INTEGER
                        attr_type_str = 'integer'                        

                    elif attr_type_str.startswith('numeric'):
                        attr_type = ATTR_NUMERIC
                        attr_type_str = 'numeric'

                    elif attr_type_str.startswith('string'):
                        attr_type = ATTR_STRING
                        attr_type_str = 'string'
                        raise NotImplementedError("STRING type attributes are not supported yet.")

                    elif attr_type_str.startswith('date'):
                        attr_type = ATTR_DATE
                        attr_type_str = 'date'
                        raise NotImplementedError("DATE type attributes are not supported yet.")

                    elif attr_type_str.startswith('relational'):
                        attr_type = ATTR_RELATIONAL
                        attr_type_str = 'relational'
                        raise NotImplementedError("RELATIONAL type attributes are not supported yet.")

                    else:
                        raise ValueError("Unknown attribute - %s" % attr_type_str)

                    # Record the type and value (if any)
                    n_attributes += 1

                    categories_i = new cpp_hashmap[string, size_t]()
                    # Index categories from 1, as default key-not-found return value is 0
                    for i, category in enumerate(category_list, start=1):
                        # 0 indexing to dereference the pointer
                        (categories_i[0])[<bytes>category] = i
                    
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
                    self.relation = line[9:].strip()

                elif first_field.startswith('@data'):
                    metadata_read = True
                    self.n_attributes = n_attributes
                    # The data_line_no is the 0-indexed number of the line at which
                    # the "@DATA" meta header is found
                    self.data_header_line_no = line_no
                    # print "Data at", line_no
                    break
                    
    cdef _scan_data(self):
        cdef size_t n_samples = 0
        cdef data_header_line_no = self.data_header_line_no
        cdef bint is_data_sparse = self.is_data_sparse

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
            if not is_data_sparse and first_char == 123:
                is_data_sparse = True
                
            n_samples += 1
        
        self.is_data_sparse = is_data_sparse
        self.n_samples = n_samples

        free(line)    # Deallocate the line buffer
        fclose(cfile) # Close the file stream
    
    cdef _read_data_dense(self, DTYPE_t[:, :] data):
        cdef size_t n_samples = self.n_samples
        cdef size_t n_attributes = self.n_attributes
        
        cdef data_header_line_no = self.data_header_line_no
        cdef bint is_data_sparse = self.is_data_sparse

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
                
            print "Parsing data line", line,
            
            # Clear attr buffer variables
            attr_len = 0
            inside_quotes = False
            inside_attr_field = False
            
            unparsed_attr = False
            col = 0   # Attr number
            char_index = 0
            
            while char_index < line_len:
                # Process non-attribute characters - ,  '  \n  \r \0
                # To speed up, order the checks based on their frequency
                
                ch = line[char_index]
                
                # Skip all spaces that are not enclosed within quotes
                if ch == 32:
                    # Do not skip spaces inside quotes
                    if not inside_quotes:
                        while line[char_index] == 32:
                            char_index += 1
                        continue
                        
                elif ch == 39:  # --> ' (quote)
                    if not inside_quotes:
                        inside_quotes = True
                        inside_attr_field = True
                        unparsed_attr = True
                        char_index += 1
                        # Don't continue go to end and parse the attr
                        
                    # If inside quotes
                    else:
                        inside_quotes = False
                        # This quote terminates the attribute field
                        # Now lets skip all chars until the next comma
                        inside_attr_field = False
                        # Skip the quote character
                        char_index += 1
                        continue

                # non-space char in/out quotes or space inside quotes
                elif ch == 44: # --> , (comma)
                    # If its a comma and we have already done processing the current attr
                    if not inside_attr_field:
                        if not unparsed_attr:
                            # Increment to next char and next attr
                            char_index += 1
                            continue
                        
                    # If we aren't done processing the current attribute
                    else:
                        # Note that we've done storing the current attribute
                        # and that we need to process it.
                        inside_attr_field = False    
                        char_index += 1
                        # Don't continue... go to the end and process the attr_buffer
                
                
                # Have to check both '\n' and '\r' for OS portability.
                elif ch == 10 or ch == 13 or ch == 0:
                    # Break if EOL '\n' or '\r' or '\0'
                    if not inside_attr_field:
                        break
                        
                    # If we reached the EOL and there is an unparsed attr (last col)
                    # Don't break        
                    else:
                        # Note that we've done storing the current attribute
                        # and that we need to process it.
                        inside_attr_field = False                    
                
                # It reaches here if - 
                #   it is a space inside a quote
                #   non-space non-meta chars inside/outside quotes
                # We need to store this in the attribute buffer
                else:
                    attr_buffer[attr_len] = ch
                    attr_len += 1
                    char_index += 1
                    
                    # If unparsed_attr is set to False, inside_attr_field is
                    # also False
                    # But we can have a parsed attribute and incomplete attribute field
                    # (spaces inside the attribute field)
                    if not unparsed_attr:
                        unparsed_attr = True
                        inside_attr_field = True
                    continue
                        
                # Now if we have unparsed attribute and we are not inside attribute field
                if not inside_attr_field and unparsed_attr:
                    # parse and store the attribute to the data matrix
                    # Based on the type of the current attribute column
                    
                    print attr_buffer, row, col,
                    
                    attr_type = (self.attribute_types[0])[col]
                    
                    # If missing value store nan, no matter what the type is
                    if attr_buffer[0] == 63:
                        print "missing"
                        data[row, col] = np.nan
                        
                    elif (attr_type == ATTR_NUMERIC or
                              attr_type == ATTR_INTEGER or
                              attr_type == ATTR_REAL):
                        print "converting to float"
                        data[row, col] = string_to_double(attr_buffer, attr_len)

                    elif attr_type == ATTR_NOMINAL:
                        # End the attr string with a NULL terminator and get the integer code
                        print "converting to category index"
                        attr_buffer[attr_len] = 0
                        data[row, col] = (self.categories[0])[col][<string>attr_buffer]

                    # Clear the attr variable
                    col += 1
                    attr_len = 0
                    unparsed_attr = False
                             
            # Parse the next line
            row += 1

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
        # Gets the n_samples, is_data_sparse, missing_exist
        self._scan_data()
        
        meta_info = dict()
        meta_info['relation'] = self.relation
        meta_info['filename'] = self.filename
        meta_info['n_attrbutes'] = self.n_attributes
        
        all_attrs = dict()
        
        for i in range(self.n_attributes):
            attr_name = (self.attribute_names[0])[i]
            
            this_attr = dict()
            this_attr['order'] = i
            
            this_attr['type'] = (self.attribute_types_str[0])[i]
            this_attr['type_index'] = (self.attribute_types[0])[i]
            
            this_attr['is_categorical'] = (self.categorical[0])[i]
            this_attr['n_categories'] = (self.n_categories[0])[i]
            this_attr['category_index_map'] = dict([cat for cat in ((self.categories[0])[i])])
            
            all_attrs[attr_name] = this_attr
            
        meta_info['attributes'] = all_attrs
        meta_info['is_sparse'] = self.is_data_sparse
        meta_info['missing_exist'] = self.missing_exist
        meta_info['encode_nominal_values_to_int'] = self.encode_labels
        
        meta_info['data_header_line_no'] = self.data_header_line_no
        meta_info['n_samples'] = self.n_samples
        
        return meta_info
    
    def get_data(self):
        data_shape = (self.n_samples, self.n_attributes)
        
        if self.is_data_sparse:
            data = csr_matrix(data_shape, dtype=np.float64)
            self._read_data_sparse(data)
            
        else:
            data = np.empty(data_shape, dtype=np.float64)
            self._read_data_dense(data)

        return data
