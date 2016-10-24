import numpy as np
import dataset_utils as du


def load_1000_genomes_hist(transpose=False, label_splits=None,
                           feature_splits=None, fold=0, perclass=False,
                           norm=True):

    train, valid, test, _ = du.load_1000_genomes(transpose, label_splits,
                                                 feature_splits, fold,
                                                 norm=False)

    # Generate no_label: fuse train and valid sets
    nolabel_orig = (np.vstack([train[0], valid[0]])).transpose()
    nolabel_y = np.vstack([train[1], valid[1]])

    nolabel_y = nolabel_y.argmax(axis=1)

    if perclass:
        nolabel_x = np.zeros((nolabel_orig.shape[0], 3*26))
        for i in range(nolabel_x.shape[0]):
            for j in range(nolabel_x.shape[1]):
                nolabel_x[i, nolabel_y[j]*3:nolabel_y[j]*3+3] += \
                    np.bincount(nolabel_orig[i, :].astype('int32'), minlength=3)
    else:
        nolabel_x = np.zeros((nolabel_orig.shape[0], 3))
        for i in range(nolabel_x.shape[0]):
            nolabel_x[i, :] += np.bincount(nolabel_orig[i, :].astype('int32'),
                                           minlength=3)

    if norm:
        train[0] = (train[0] - train[0].mean(axis=0)[None, :]) / \
            train[0].std(axis=0)[None, :]
        valid[0] = (valid[0] - valid[0].mean(axis=0)[None, :]) / \
            valid[0].std(axis=0)[None, :]
        test[0] = (test[0] - test[0].mean(axis=0)[None, :]) / \
            test[0].std(axis=0)[None, :]

    nolabel_x = nolabel_x.astype('float32')

    import ipdb; ipdb.set_trace()

    return train, valid, test, nolabel_x

if __name__ == '__main__':
    train, valid, test, nolabel_x = load_1000_genomes_hist(
        transpose=False, label_splits=[.75], feature_splits=None, fold=0,
        perclass=True, norm=True)
