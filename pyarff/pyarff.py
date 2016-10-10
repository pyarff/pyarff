# Authors: Raghav R V <rvraghav93@gmail.com>
#
# Licence: BSD 3 clause

import numpy as np
from ._pyarff import ARFFReader

__all__ = ["load_arff_dataset",]

def load_arff_dataset(filename, load_data=True, encode_nominals=True):
    data = None

    arffreader = ARFFReader(filename=bytes(filename, 'utf-8'),
                            encode_nominals=encode_nominals)
    meta_data = arffreader.get_metadata()

    if load_data:
        data = arffreader.get_data()

    return meta_data, data
