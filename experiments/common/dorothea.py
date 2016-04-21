import numpy
from collections import OrderedDict

from fuel.datasets import Dataset, IndexableDataset
from config import path_dorothea

def load_data(which, mode='standard', shuffle=False, return_format='fuel'):

    assert which in ['train', 'valid', 'test']
    assert mode in ['standard', 'feature_representation']
    assert return_format in ['numpy', 'fuel']

    train_x, train_y, valid_x, valid_y, test_x = read_files()

    if which == 'train':
        # indices = numpy.concatenate((numpy.argwhere(train_y==0)[:10],
        #                             numpy.argwhere(train_y==1)[:10]),
        #                            axis=0)[:,0]
        x = train_x
        y = train_y
    elif which == 'valid':
        # indices = numpy.concatenate((numpy.argwhere(valid_y==0)[:10],
        #                             numpy.argwhere(valid_y==1)[:10]),
        #                            axis=0)[:,0]
        x = valid_x
        y = valid_y

    elif which == 'test':
        x = test_x

        # NOTE : The labels from the training set are used to simplify the
        # following transformation code but they are NOT returned to the user.
        y = train_y

    # Shuffle x and y together, if needed
    if shuffle:
        p = numpy.random.permutation(len(x))
        x = x[p]
        y = y[p]

    if mode == 'feature_representation':
        example_indx = numpy.arange(x.shape[0]).repeat(x.shape[1])
        feature_indx = numpy.tile(numpy.arange(x.shape[1]), x.shape[0])

        y = y.repeat(x.shape[1])
        x = numpy.concatenate((example_indx[:,None], feature_indx[:, None]),
                              axis=1)

    # Format x and y into a Fuel Indexable dataset
    if return_format == "numpy":
        if which == "test":
            return x
        else:
            return x, y
    elif return_format == "fuel":
        if which == "test":
            return IndexableDataset(indexables=OrderedDict([('features', x)]),
                                    axis_labels={'features': ('batch',
                                                              'length')},
                                    sources=('features'))
        else:
            return IndexableDataset(indexables=OrderedDict([('features', x),
                                                            ('targets', y)]),
                                    axis_labels={'features': ('batch',
                                                              'length'),
                                                 'targets': ('batch',)},
                                    sources=('features', 'targets'))


def read_files():
    ntrain = 800
    nvalid = 350
    ntest = 800

    dim = 100000

    def do_x(fn, n):
        mtx = numpy.zeros((n, dim), dtype=numpy.float32)
        x = 0
        for x, l in enumerate(open(fn)):
            for y in l.rstrip().split(' '):
                mtx[x, int(y)-1] = 1
        return mtx

    train_x = do_x(path_dorothea+'dorothea_train.data', ntrain)
    train_y = numpy.equal(numpy.loadtxt(path_dorothea+'dorothea_train.labels'), 1)

    valid_x = do_x(path_dorothea+'dorothea_valid.data', nvalid)
    valid_y = numpy.equal(numpy.loadtxt(path_dorothea+'dorothea_valid.labels'), 1)

    test_x = do_x(path_dorothea+'dorothea_test.data', ntest)

    return train_x, train_y, valid_x, valid_y, test_x
