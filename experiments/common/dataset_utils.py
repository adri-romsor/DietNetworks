import numpy
import os

from feature_selection.experiments.common import (protein_loader, dorothea,
                                                  reuters, imdb, iric_molecules,
                                                  thousand_genomes)
from feature_selection import aggregate_dataset as opensnp


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


def load_protein_binding(transpose=False, splits=None):
    x, y = protein_loader.load_data()
    y = y[:, None]

    if transpose:
        x = x.transpose()
        split_data = split([x], splits)
    else:
        x, y = shuffle([x, y])
        split_data = split([x, y], splits)

    return split_data


def load_dorothea(transpose=False, splits=None):
    # WARNING : Temporary solution : use the valid set as a test set because
    # dorothea has no labels for the test set
    train = dorothea.load_data('train', 'standard', False, 'numpy')
    valid = dorothea.load_data('valid', 'standard', False, 'numpy')

    train = (train[0], train[1][:, None])
    valid = (valid[0], valid[1][:, None])

    if transpose:
        all_x = numpy.vstack(train[0], valid[0]).transpose()
        return split([all_x], splits)
    else:
        if splits is not None:
            raise AssertionError("Standard training, validation and test sets "
                                 "are already defined for the Dorothea dataset")
        assert splits is None

        train = shuffle(train)
        valid = shuffle(valid)
        return train, valid, valid, None


def load_opensnp(transpose=False, splits=None):

    # Load all of the data, separating the unlabeled data from the labeled data
    data = opensnp.load_data23andme_baselines(split=1.0)
    (x_sup, x_sup_labels), _, x_unsup = data
    x_sup_labels = x_sup_labels[:, None]

    # Cast the data to the right dtype
    x_sup = x_sup.astype("float32")
    x_sup_labels = x_sup_labels.astype("float32")
    x_unsup = x_unsup.astype("float32")

    if transpose:
        all_x = numpy.vstack((x_sup, x_unsup)).transpose()
        return split([all_x], splits)
    else:
        # Separate the labeled data into train, valid and test
        (x_sup, x_sup_labels) = shuffle((x_sup, x_sup_labels))
        train, valid, test = split([x_sup, x_sup_labels], splits)
        return train, valid, test, x_unsup


def load_reuters(transpose=False, splits=None):

    train, test = reuters.ReutersDataset().load_data()

    if transpose:
        all_x = train[0].transpose()
        return split([all_x], splits)
    else:
        if splits is not None and len(splits) > 1:
            print("A test set is already defined for the Reuters dataset. "
                  "Therefore the last requested split will be ignored and the "
                  "previous splits will be renormalized")
            splits = prune_splits(splits, nb_prune=1)

        train = shuffle(train)
        train, valid = split(train, splits)
        return train, valid, test, None


def load_iric_molecules(transpose=False, splits=None):

    d = iric_molecules.IricMoleculesDataset()
    x = d.load_data("fingerprint")
    y = d.load_data("affinity")

    if transpose:
        return split([x.transpose()], splits)
    else:
        (x, y) = shuffle((x, y))
        train, valid, test = split([x, y], splits)
        return train, valid, test, None


def load_1000_genomes(transpose=False, label_splits=None, feature_splits=None,
                      fold=0):

    # user = os.getenv("USER")
    path = "/data/lisatmp4/romerosa/datasets/1000_Genome_project/"  # % user
    x, y = thousand_genomes.load_data(path)
    x = x.astype("float32")

    x = (x - x.mean(axis=0)[None, :]) / x.std(axis=0)[None, :]

    (x, y) = shuffle((x, y))

    if transpose:
        # This will only be used for unsupervised pretraining
        assert len(feature_splits) == 1# train/valid split
        train, valid = split([x, y], label_splits)
        (transposed,) = shuffle((train[0].transpose(),))
        return split([train[0].transpose()], feature_splits)
    else:
        # This will only be used for supervised training
        assert len(label_splits) == 1# train/valid split
        # 5-fold cross validation: this means that test will always be 20%
        all_folds = split([x, y], [.2, .2, .2, .2])
        assert fold >= 0
        assert fold < 5

        # Separate choosen test set
        test = all_folds[fold]
        all_folds = all_folds[:fold] + all_folds[(fold + 1):]

        # Prepare training and validation sets
        x = numpy.concatenate([el[0] for el in all_folds])
        y = numpy.concatenate([el[1] for el in all_folds])
        train, valid = split([x, y], label_splits)
        return train, valid, test, None

def load_imdb(transpose=False, splits=None, unlabeled=False, shuffle=False):
    '''
    loads imdb dataset so that it can fits in memory for each single
    case of interest
    '''
    if transpose:
        if unlabeled:
            train_data, _, unlab_data, _ = imdb.load_imdb_BoW(shuffle=shuffle)
            all_x = numpy.empty((train_data.shape[0]+unlab_data.shape[0],
                                 train_data.shape[1]), dtype='float32')
            all_x_t = all_x[:train_data.shape[0]]
            all_x_unl = all_x[train_data.shape[0]:]
            all_x_t[:] = train_data.toarray().astype("float32")
            all_x_unl[:] = unlab_data.toarray().astype("float32")
        else:
            train_data, _, _, _ = imdb.load_imdb_BoW(shuffle=shuffle)
            train_data = train_data.toarray().astype("float32")
            all_x = train_data
        all_x = all_x.transpose()
        return split([all_x], splits)
    else:
        # separate labeled data into train and val
        train_data, train_labels, _,\
            test_data = imdb.load_imdb_BoW(shuffle=shuffle)
        if splits is not None and len(splits) > 1:
            splits = prune_splits(splits, nb_prune=1)
        train_x = train_data.toarray().astype("float32")
        train_labels = train_labels.astype("int32")
        test_data = test_data.toarray().astype("float32")
        train, val = split([train_x, train_labels], splits=splits)
        return train, val, test_data, unlab_data
