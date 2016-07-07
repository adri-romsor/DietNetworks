import os
from sklearn.feature_extraction.text import CountVectorizer
from Word2VecUtility import Word2VecUtility
import nltk.data
import logging
from gensim.models import Word2Vec
import pandas as pd
import numpy as np


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
    train_data_features = train_data_features.toarray()
    train_labels = np.array(train['sentiment'])
    test_data_features = test_data_features.toarray()

    return train_data_features, train_labels, unlab_data_features,\
        test_data_features


def load_imdb_BoW(path_to_files='/data/lisatmp4/erraqabi/data/imdb_reviews/',
                  shuffle=False, seed=0):

    data = np.load(os.path.join(path_to_files, 'imdb.npz'))
    train_data_features = data['train_data_features']
    train_labels = data['train_labels']
    unlab_data_features = data['unlab_data_features']
    test_data_features = data['test_data_features']

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
    np.savez('imdb.npz', train_data_features=train_data_features,
             train_labels=train_labels,
             test_data_features=test_data_features,
             unlab_data_features=unlab_data_features)

if __name__ == '__main__':
    process_and_save_imdb()
