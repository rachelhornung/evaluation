__author__ = 'korhammer'

import numpy as np


def based_floor(x, base=10):
    """
    rounds down to the last integer of base 10 (or other)
    """
    return np.int(base * np.floor(np.float(x) / base))


def based_ceil(x, base=10):
    """
    rounds up to the next integer of base 10 (or other)
    """
    return np.int(base * np.ceil(np.float(x) / base))


def float(x):
    return np.float(x) if len(x) > 0 else np.nan


class LabelEncoderResorted():
    '''
    encodes labels without sorting them
    '''

    def __init__( self ):
        self.classes_ = None

    def fit( self, classes, order=None ):
        if order == 'ascend':
            # sort ascending
            self.classes_ = np.unique( classes ).tolist()
        elif order == None:
            indexes = np.unique( classes, return_index=True )[1]
            self.classes_ = [classes[index] for index in sorted( indexes )]
        elif np.all( np.unique( classes ) == np.unique( order ) ):
            # sort as given in order
            indexes = np.unique( order, return_index=True )[1]
            self.classes_ = [order[index] for index in sorted( indexes )]
        else:
            raise AttributeError( 'order not correct' )

    def transform( self, labels ):
        transformed = []
        for l in labels:
            transformed.append( self.classes_.index( l ) )
        return transformed

    def fit_transform( self, labels ):
        self.fit( labels )
        return self.transform( labels )

    def inverse_transform( self, labels ):
        transformed = []
        for l in labels:
            transformed.append( self.classes_[l] )
        return transformed