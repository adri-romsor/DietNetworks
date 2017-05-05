from __future__ import print_function
import numpy
import os
"""
from DietNetworks.experiments.common import (protein_loader, dorothea,
                                                  reuters, imdb, iric_molecules,
                                                  thousand_genomes)
"""
from DietNetworks.experiments.common import thousand_genomes

from DietNetworks import aggregate_dataset as opensnp


def shuffle(data_sources, seed=23):
    """
    Shuffles multiple data sources (numpy arrays) together so the
    correspondance between data sources (such as inputs and targets) is
    maintained.
    """

    numpy.random.seed(seed)
    indices = numpy.arange(data_sources[0].shape[0])
    numpy.random.shuffle(indices)

    return [d[indices] for d in data_sources]


def split(data_sources, splits):
    """
    Splits the given data sources (numpy arrays) according to the provided
    split boundries.

    Ex : if splits is [0.6], every data source will be separated in two parts,
    the first containing 60% of the data and the other containing the
    remaining 40%.
    """

    if splits is None:
        return data_sources

    split_data_sources = []
    nb_elements = data_sources[0].shape[0]
    start = 0
    end = 0

    for s in splits:
        end += int(nb_elements * s)
        split_data_sources.append([d[start:end] for d in data_sources])
        start = end
    split_data_sources.append([d[end:] for d in data_sources])

    return split_data_sources


def prune_splits(splits, nb_prune):
    """
    Takes as input a list of split points in a dataset and produces a new list
    where the last split has been removed and the other splits are expanded
    to encompass the whole data but keeping the same proportions.

    Ex : splits = [0.6, 0.2] corresponds to 3 splits : 60%, 20% and 20%.
         with nb_prune=1, the method would return [0.75] which corresponds to
         2 splits : 75% and 25%. Hence, the proportions between the remaining
         splits remain the same.
    """
    if nb_prune > 1:
        normalization_constant = (1.0 / sum(splits[:-(nb_prune-1)]))
    else:
        normalization_constant = 1.0 / sum(splits)
    return [s * normalization_constant for s in splits[:-nb_prune]]


def load_1000_genomes(transpose=False, label_splits=None, feature_splits=None,
                      nolabels='raw', fold=0, norm=True,
                      path="/data/lisatmp4/romerosa/datasets/1000_Genome_project/" ):

    # user = os.getenv("USER")
    print(path)

    if nolabels == 'raw' or not transpose:
        # Load raw data either for supervised or unsupervised part
        x, y = thousand_genomes.load_data(path)
        x = x.astype("float32")

        (x, y) = shuffle((x, y))  # seed is fixed, shuffle is always the same

        # Prepare training and validation sets
        assert len(label_splits) == 1  # train/valid split
        # 5-fold cross validation: this means that test will always be 20%
        all_folds = split([x, y], [.2, .2, .2, .2])
        assert fold >= 0
        assert fold < 5

        # Separate choosen test set
        test = all_folds[fold]
        all_folds = all_folds[:fold] + all_folds[(fold + 1):]

        x = numpy.concatenate([el[0] for el in all_folds])
        y = numpy.concatenate([el[1] for el in all_folds])

    # Data used for supervised training
    if not transpose:
        train, valid = split([x, y], label_splits)
        if norm:
            mu = x.mean(axis=0)
            sigma = x.std(axis=0)
            train[0] = (train[0] - mu[None, :]) / sigma[None, :]
            valid[0] = (valid[0] - mu[None, :]) / sigma[None, :]
            test[0] = (test[0] - mu[None, :]) / sigma[None, :]
        rvals = [train, valid, test]
    else:
        rvals = []

    # Data used for transpose part or unsupervised training
    if nolabels == 'raw' and not transpose:
        unsupervised_data = None  # x.transpose()
    elif nolabels == 'raw' and transpose:
        unsupervised_data = x.transpose()
    elif nolabels == 'histo3':
        unsupervised_data = numpy.load(os.path.join(path, 'histo3_fold' +
                                    str(fold) + '.npy'))
    elif nolabels == 'histo3x26':
        unsupervised_data = numpy.load(os.path.join(path, 'histo3x26_fold' +
                                    str(fold) + '.npy'))
    elif nolabels == 'bin':
        unsupervised_data = numpy.load(os.path.join(path, 'snp_bin_fold' +
                                        str(fold) + '.npy'))
    elif nolabels == 'w2v':
        raise NotImplementedError
    else:
        try:
            unsupervised_data = numpy.load(nolabels)
        except:
            raise ValueError('Could not load specified embedding source')

    if transpose:
        assert len(feature_splits) == 1  # train/valid split feature-wise
        (unsupervised_data, ) = shuffle((unsupervised_data,))
        rvals += split([unsupervised_data], feature_splits)
    else:
        rvals += [unsupervised_data]

    return rvals
