import os
import numpy as np
import dataset_utils as du


def generate_1000_genomes_hist(transpose=False, label_splits=None,
                               feature_splits=None, fold=0, perclass=False):

    train, valid, test, _ = du.load_1000_genomes_old(transpose, label_splits,
                                                     feature_splits, fold,
                                                     norm=False)

    # Generate no_label: fuse train and valid sets
    nolabel_orig = (np.vstack([train[0], valid[0]])).transpose()
    nolabel_y = np.vstack([train[1], valid[1]])

    nolabel_y = nolabel_y.argmax(axis=1)

    path = '/data/lisatmp4/romerosa/datasets/1000_Genome_project/'

    filename = 'unsupervised_hist_3x26' if perclass else \
        'unsupervised_hist_3'
    filename += '_fold' + str(fold) + '.npy'

    if perclass:
        nolabel_x = np.zeros((nolabel_orig.shape[0], 3*26))
        for i in range(nolabel_x.shape[0]):
            for j in range(nolabel_x.shape[1]):
                nolabel_x[i, nolabel_y[j]*3:nolabel_y[j]*3+3] += \
                    np.bincount(nolabel_orig[i, :].astype('int32'),
                                minlength=3)
    else:
        nolabel_x = np.zeros((nolabel_orig.shape[0], 3))
        for i in range(nolabel_x.shape[0]):
            nolabel_x[i, :] += np.bincount(nolabel_orig[i, :].astype('int32'),
                                           minlength=3)

    nolabel_x = nolabel_x.astype('float32')

    np.save(os.path.join(path, filename), nolabel_x)


def generate_1000_genomes_snp2bin(transpose=False, label_splits=None,
                                  feature_splits=None, fold=0):

    train, valid, test, _ = du.load_1000_genomes_old(transpose, label_splits,
                                                     feature_splits, fold,
                                                     norm=False)

    # Generate no_label: fuse train and valid sets
    nolabel_orig = (np.vstack([train[0], valid[0]]))
    nolabel_x = np.zeros((nolabel_orig.shape[0], nolabel_orig.shape[1]*2),
                         dtype='uint8')

    path = '/data/lisatmp4/romerosa/datasets/1000_Genome_project/'
    filename = 'unsupervised_snp_bin_fold' + str(fold) + '.npy'

    # SNP to bin
    nolabel_x[:, ::2] += (nolabel_orig == 2)
    nolabel_x[:, 1::2] += (nolabel_orig >= 1)
    
    np.save(os.path.join(path, filename), nolabel_x)


if __name__ == '__main__':

    for f in range(5):
        print(str(f))
        generate_1000_genomes_snp2bin(
            transpose=False, label_splits=[.75], feature_splits=[.8], fold=f)
