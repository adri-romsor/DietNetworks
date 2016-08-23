import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from Word2VecUtility import Word2VecUtility
import nltk.data
import logging
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import h5py
import tables

import argparse


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph

    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")

    nwords = 0.

    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)

    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])

    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array

    # Initialize a counter
    counter = 0.
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
        # Print a status message every 1000th review
        if counter % 1000. == 0.:
            print "Review %d of %d" % (counter, len(reviews))

        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(review, model,
                                                    num_features)
        # Increment the counter
        counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append(Word2VecUtility.review_to_wordlist(
            review, remove_stopwords=True))
    return clean_reviews


def build_imdb_BoW(path_to_data='/data/lisatmp4/erraqabi/data/imdb_reviews/',
                   max_features=None, use_unlab=True):
    # load data
    train = pd.read_csv(os.path.join(path_to_data,
                                     'labeledTrainData.tsv'),
                        header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(path_to_data,
                                    'testData.tsv'),
                       header=0, delimiter="\t", quoting=3)
    unlabeled_data = pd.read_csv(os.path.join(path_to_data,
                                              'unlabeledTrainData.tsv'),
                                 header=0, delimiter="\t", quoting=3)

    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []
    clean_test_reviews = []
    clean_unlab_train_reviews = []

    # cleaning the reviews text
    print "Cleaning and parsing the training set movie reviews...\n"
    for i in xrange(0, len(train["review"])):
        clean_train_reviews.append(" ".join(
            Word2VecUtility.review_to_wordlist(
                train["review"][i], True)))
    for i in xrange(0, len(test["review"])):
        clean_test_reviews.append(" ".join(
            Word2VecUtility.review_to_wordlist(
                test["review"][i], True)))
    for i in xrange(0, len(unlabeled_data["review"])):
        clean_unlab_train_reviews.append(" ".join(
            Word2VecUtility.review_to_wordlist(
                unlabeled_data["review"][i], True)))

    print "Creating the bag of words...\n"
    # Initialize the vectorizer
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=max_features)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors.
    if use_unlab:
        vectorizer.fit(clean_train_reviews+clean_unlab_train_reviews)
        train_data_features = vectorizer.transform(clean_train_reviews)
    else:
        train_data_features = vectorizer.fit_transform(clean_train_reviews)

    unlab_data_features = vectorizer.transform(clean_unlab_train_reviews)
    # For the test set, we use the same vocab as for the train set
    test_data_features = vectorizer.transform(clean_test_reviews)

    # convert the result to an array
    # train_data_features = train_data_features
    train_labels = np.array(train['sentiment'])
    # test_data_features = test_data_features

    return train_data_features, train_labels, unlab_data_features,\
        test_data_features


def build_imdb_tfidf(path_to_data='/data/lisatmp4/erraqabi/data/imdb_reviews/',
                     max_features=None, use_unlab=True):
    # load data
    train = pd.read_csv(os.path.join(path_to_data,
                                     'labeledTrainData.tsv'),
                        header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(path_to_data,
                                    'testData.tsv'),
                       header=0, delimiter="\t", quoting=3)
    unlabeled_data = pd.read_csv(os.path.join(path_to_data,
                                              'unlabeledTrainData.tsv'),
                                 header=0, delimiter="\t", quoting=3)

    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []
    clean_test_reviews = []
    clean_unlab_train_reviews = []

    # cleaning the reviews text
    print "Cleaning and parsing the training set movie reviews...\n"
    for i in xrange(0, len(train["review"])):
        clean_train_reviews.append(" ".join(
            Word2VecUtility.review_to_wordlist(
                train["review"][i], True)))
    for i in xrange(0, len(test["review"])):
        clean_test_reviews.append(" ".join(
            Word2VecUtility.review_to_wordlist(
                test["review"][i], True)))
    for i in xrange(0, len(unlabeled_data["review"])):
        clean_unlab_train_reviews.append(" ".join(
            Word2VecUtility.review_to_wordlist(
                unlabeled_data["review"][i], True)))

    print "Creating the bag of words...\n"
    # Initialize the vectorizer
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=max_features)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors.
    if use_unlab:
        vectorizer.fit(clean_train_reviews+clean_unlab_train_reviews)
        train_data_features = vectorizer.transform(clean_train_reviews)
    else:
        train_data_features = vectorizer.fit_transform(clean_train_reviews)
    # build the tf-idf transformer
    tf_transformer = TfidfTransformer(use_idf=False).fit(train_data_features)
    # transform the counts to tf-idf
    train_data_features = tf_transformer.transform(train_data_features)
    unlab_data_features = tf_transformer.transform(
        vectorizer.transform(clean_unlab_train_reviews))
    # For the test set, we use the same vocab as for the train set
    test_data_features = tf_transformer.transform(
        vectorizer.transform(clean_test_reviews))

    # convert the result to an array
    # train_data_features = train_data_features
    train_labels = np.array(train['sentiment'])
    # test_data_features = test_data_features

    return train_data_features, train_labels, unlab_data_features,\
        test_data_features


def load_imdb(path_to_files='/data/lisatmp4/erraqabi/data/imdb_reviews/',
              feat_type='BoW', shuffle=False, seed=0):
    data = np.load(os.path.join(path_to_files, 'imdb_'+feat_type+'.npz'))
    train_data_features = data['train_data_features'].item()
    train_labels = data['train_labels']
    unlab_data_features = data['unlab_data_features'].item()
    test_data_features = data['test_data_features'].item()

    if shuffle:
        np.random.seed(seed)
        idx_shuffle = np.random.permutation(train_data_features.shape[0])
        train_data_features = train_data_features[idx_shuffle]
        train_labels = train_labels[idx_shuffle]
        idx_shuffle = np.random.permutation(test_data_features.shape[0])
        test_data_features = test_data_features[idx_shuffle]
        idx_shuffle = np.random.permutation(unlab_data_features.shape[0])
        unlab_data_features = unlab_data_features[idx_shuffle]

    return train_data_features, train_labels, unlab_data_features,\
        test_data_features


def load_imdb_word2vec(path_to_data, model_path=None, use_unlab=True):
    # load data
    train = pd.read_csv(os.path.join(path_to_data,
                                     'labeledTrainData.tsv'),
                        header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(path_to_data,
                                    'testData.tsv'),
                       header=0, delimiter="\t", quoting=3)
    unlabeled_data = pd.read_csv(os.path.join(path_to_data,
                                              'unlabeledTrainData.tsv'),
                                 header=0, delimiter="\t", quoting=3)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = []  # Initialize an empty list of sentences

    print "Parsing sentences from training set"
    for review in train["review"]:
        sentences += Word2VecUtility.review_to_sentences(review, tokenizer)
    if use_unlab:
        print "Parsing sentences from unlabeled set"
        for review in unlabeled_data["review"]:
            sentences += Word2VecUtility.review_to_sentences(
                review, tokenizer)

    # load model from path if provided
    if model_path is not None:
        model = Word2Vec.load(model_path)
        num_features = len(model['human'])
    else:
        # Set values for various parameters
        num_features = 300    # Word vector dimensionality
        min_word_count = 40   # Minimum word count
        num_workers = 4       # Number of threads to run in parallel
        context = 10          # Context window size
        downsampling = 1e-3   # Downsample setting for frequent words

        # Initialize and train the model (this will take some time)
        print "Training Word2Vec model..."
        model = Word2Vec(sentences, workers=num_workers,
                         size=num_features, min_count=min_word_count,
                         window=context, sample=downsampling, seed=1)
        model_name = str(num_features)+'features_' + \
            str(min_word_count)+'minwords_'+str(context)+'context'
        model.save(model_name)

    # TODO: add weighted average feature, make feature type a parameter
    # if feat_type = 'average':

    # Create average vectors for the training and test sets
    print "Creating feature vecs for training reviews"
    train_data_features = getAvgFeatureVecs(getCleanReviews(train), model,
                                            num_features)
    train_labels = np.array(train['sentiment'])
    print "Creating average feature vecs for test reviews"
    test_data_features = getAvgFeatureVecs(getCleanReviews(test), model,
                                           num_features)

    return train_data_features, train_labels, test_data_features


def build_and_save_imdb(path='/data/lisatmp4/erraqabi/data/imdb_reviews/',
                        feat_type='BoW'):
    if feat_type == 'BoW':
        train_data_features, train_labels, unlab_data_features,\
            test_data_features = build_imdb_BoW()
    if feat_type == 'tfidf':
        train_data_features, train_labels, unlab_data_features,\
            test_data_features = build_imdb_tfidf()
    np.savez(os.path.join(path, 'imdb_'+feat_type+'.npz'),
             train_data_features=train_data_features,
             train_labels=train_labels,
             test_data_features=test_data_features,
             unlab_data_features=unlab_data_features)


def save_as_hdf5(path='/Tmp/erraqaba/datasets/imdb/', unsupervised=True,
                 feat_type='BoW', use_tables=True, split=0.8):
    if not os.path.exists(path):
        os.makedirs(path)
    if unsupervised:
        train_data, _, unlab_data, _ = load_imdb(feat_type=feat_type)
        if use_tables:
            f = tables.openFile(os.path.join(path,
                                             'unsupervised_IMDB_'+feat_type +
                                             '_table'
                                             '_split80.hdf5'),
                                mode='w')
            features = np.empty((train_data.shape[1],
                                 train_data.shape[0]+unlab_data.shape[0]),
                                dtype='float32')
            x_t = features.transpose()[:train_data.shape[0]]
            x_unl = features.transpose()[train_data.shape[0]:]
            x_t[:] = train_data.toarray().astype("float32")
            x_unl[:] = unlab_data.toarray().astype("float32")
            f.createArray(f.root, 'train',
                          features[:int(features.shape[0]*split)])
            f.createArray(f.root, 'val',
                          features[int(features.shape[0]*split):])
        else:
            f = h5py.File(os.path.join(path, 'unsupervised_IMDB_'+feat_type +
                                             '.hdf5'),
                          mode='w')
            features = f.create_dataset('features',
                                        (train_data.shape[1],
                                         train_data.shape[0] +
                                         unlab_data.shape[0]),
                                        dtype='float32')
            train_data, _, unlab_data, _ = load_imdb(feat_type=feat_type)
            features = np.empty((train_data.shape[1],
                                 train_data.shape[0]+unlab_data.shape[0]),
                                dtype='float32')
            x_t = features.transpose()[:train_data.shape[0]]
            x_unl = features.transpose()[train_data.shape[0]:]
            x_t[:] = train_data.toarray().astype("float32")
            x_unl[:] = unlab_data.toarray().astype("float32")
            f.flush()
    else:
        train_data, train_labels, _,\
            test_data = load_imdb(feat_type=feat_type)
        train_x = train_data.toarray().astype("float32")
        train_labels = train_labels.astype("int32")
        test_x = test_data.toarray().astype("float32")
        f = tables.openFile(os.path.join(path, 'supervised_IMDB_'+feat_type +
                                               '_table'
                                               '_split80.hdf5'),
                            mode='w')
        f.createArray(f.root, 'train_features',
                      train_x[:int(train_x.shape[0]*split)])
        f.createArray(f.root, 'val_features',
                      train_x[int(train_x.shape[0]*split):])
        f.createArray(f.root, 'train_labels',
                      train_labels[:int(train_x.shape[0]*split)])
        f.createArray(f.root, 'val_labels',
                      train_labels[int(train_x.shape[0]*split):])
        f.createArray(f.root, 'test_features', test_x)

    f.close()


def read_from_hdf5(path='/Tmp/carriepl/datasets/imdb/', unsupervised=True,
                   use_tables=True, feat_type='Bow'):
    if unsupervised:
        file_name = os.path.join(path, 'unsupervised_IMDB_' + feat_type +
                                       '_table_split80.hdf5')
    else:
        file_name = os.path.join(path, 'supervised_IMDB_' + feat_type +
                                       '_table_split80.hdf5')

    read_file = tables.open_file(file_name, mode='r')
    return read_file

if __name__ == '__main__':
    # build_and_save_imdb()

    parser = argparse.ArgumentParser(description=
            """Creating the imdb datasets""")
    parser.add_argument('--save',
                        default='/Tmp/sylvaint/datasets/imdb',
                        help='Path to save results.')
    args = parser.parse_args()
    print ("Printing args")
    print (args)

    #save_as_hdf5(unsupervised=True, use_tables=False)
    #save_as_hdf5(unsupervised=True)
    #save_as_hdf5(unsupervised=False)

    # build_and_save_imdb(feat_type='tfidf')
    save_as_hdf5(unsupervised=True, feat_type='tfidf', use_tables=False)
    save_as_hdf5(unsupervised=True, feat_type='tfidf')
    save_as_hdf5(unsupervised=False, feat_type='tfidf')
