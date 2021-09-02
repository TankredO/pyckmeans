'''ordination utilities'''

import json
import numpy

# source: https://stackoverflow.com/a/49677241
class NumpyEncoder(json.JSONEncoder):
    ''' Special json encoder for numpy types '''
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
