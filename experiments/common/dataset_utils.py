import numpy

from experiments.common import protein_loader
from experiments.common import dorothea
from feature selection import aggregate_dataset as opensnp


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
    
    split_data_sources = []
    nb_elements = data_sources[0].shape[0]
    start = 0
    end = 0
    
    for s in splits:
        end += int(nb_elements * split)
        split_data_sources.append([d[start:end] for d in data_sources])
        start = end
    split_data_sources.append([d[end:] for d in data_sources])    
        
    return split_data_sources
    

    
def load_protein_binding(transpose=False, splits=[0.6, 0.2]):
    x, y = protein_loader.load_data()
    assert y.ndim == 2
    
    if transpose:
        x = x.transpose()
        x, = shuffle([x])[0]
        split_data = split([x], splits)
    else:
        x, y = shuffle([x, y])
        split_data = split([x, y], splits)
    
    return split_data
    
    
def load_dorothea(transpose=False, splits=[0.6, 0.2]):
    # WARNING : Temporary solution : use the valid set as a test set because
    # dorothea has no labels for the test set
    train = dorothea.load_data('train', 'standard', False, 'numpy')
    valid = dorothea.load_data('valid', 'standard', False, 'numpy')
    assert train[1].ndim == 2
    
    if transpose:
        all_x = numpy.vstack(train[0], valid[0]).transpose()
        all_x = shuffle([all_x])[0]
        return split([all_x], splits)
    else:
        # Ignore the user defined splits, there are already defined
        assert splits is None
        
        train = shuffle(train)
        valid = shuffle(valid)
        return train, valid, valid
    
    
 def load_opensnp(transpose=False, splits=[0.6, 0.2]):

    # Load all of the data, separating the unlabeled data from the labeled data
    data = aggregate_dataset.load_data23andme_baselines(split=1.0)
    (x_sup, x_sup_labels), _, x_unsup = data
    assert x_sup_labels.ndim == 2
    
    # Cast the data to the right dtype
    x_sup = x_sup.astype("float32")
    x_sup_labels = x_sup_labels.astype("float32")
    x_unsup = x_unsup.astype("float32")
    
    if transpose:
        all_x = numpy.vstack(x_sup, x_unsup).transpose()
        all_x = shuffle([all_x])[0]
        return split([all_x], splits)
    else:
        
        # Separate the labeled data into train, valid and test
        (x_sup, x_sup_labels) = shuffle((x_sup, x_sup_labels))
        train, valid, test = split([x_sup, x_sup_labels])
        
        # Add the unlabeled data to the training data
        missing_labels = numpy.ones((len(x_unsup), 1), dtype="float32")
        x_train = numpy.vstack((train[0], x_unsup))
        y_train = numpy.vstack((train[1], missing_labels))
        x_train, y_train = shuffle((x_train, y_train))
        
        return (x_train, y_train), valid, test
    
    
 
reuters
imdb (to big for memory => store on disk)


