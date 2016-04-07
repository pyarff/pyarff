# Authors: Raghav R V <rvraghav93@gmail.com>
#
# Licence: BSD 3 clause

import numpy as np
from .pyarff import ARFFReader

__all__ = ["load_arff_dataset",]

def load_arff_dataset(filename, load_data=True, encode_labels=True):
    data = None
    
    arffreader = ARFFReader(filename=filename, encode_labels=encode_labels)
    meta_data = arffreader.get_metadata()
    
    if load_data:
        data = arffreader.get_data()        
    
    return meta_data, data
