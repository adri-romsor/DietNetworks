import os
import numpy as np


def main(embedding_path, which_method="kmeans"):

    print "Starting main loop"
    embedding_methods = []

    print "Method: {}".format(which_method)

    embedding_methods = os.listdir(embedding_path)
    embedding_methods = [emb for emb in embedding_methods
                         if ".npz" in emb and which_method in emb]

    embedding_methods = [emb for emb in embedding_methods if not
                         ("triangle" in emb)]
    embedding_methods = [emb for emb in embedding_methods if not
                         ("hard" in emb)]

    print "Embedding methods: {}".format(embedding_methods)
    mod = 1

    for embedding in embedding_methods:
        print "Training model %s: model %d out of %d" % \
                (embedding, mod, len(embedding_methods))
        f = np.load(embedding_path + embedding)

        x_train = np.array(f['x_train_supervised'], dtype=np.float32)
        y_train = np.array(f['y_train_supervised'])
        x_valid = np.array(f['x_valid_supervised'], dtype=np.float32)
        y_valid = np.array(f['y_valid_supervised'])
        x_test = np.array(f['x_test_supervised'], dtype=np.float32)
        y_test = np.array(f['y_test_supervised'])

        if x_train.min() < 0 or x_test.min() < 0:
            import pdb
            pdb.set_trace()

        new_x_train = np.zeros(x_train.shape)
        new_x_valid = np.zeros(x_valid.shape)
        new_x_test = np.zeros(x_test.shape)

        # Hard one hot encoding
        amin = x_train.argmin(1)
        new_x_train[range(new_x_train.shape[0]), amin] = 1
        amin = x_valid.argmin(1)
        new_x_valid[range(new_x_valid.shape[0]), amin] = 1
        amin = x_test.argmin(1)
        new_x_test[range(new_x_test.shape[0]), amin] = 1

        # Save new encoding
        file_name = embedding_path + embedding[:-4] + "_hard.npz"
        np.savez(file_name,
                 x_train_supervised=new_x_train,
                 y_train_supervised=y_train,
                 x_valid_supervised=x_valid,
                 y_valid_supervised=y_valid,
                 x_test_supervised=new_x_test,
                 y_test_supervised=y_test)

if __name__ == '__main__':

    embedding_path = "/data/lisatmp4/romerosa/feature_selection/1000_genomes/"
    main(embedding_path)
