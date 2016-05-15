import os
import numpy as np


def main(embedding_path, which_method="kmeans"):

    print "Starting main loop"
    embedding_methods = []

    print "Method: {}".format(which_method)

    embedding_methods = os.listdir(embedding_path)
    embedding_methods = [emb for emb in embedding_methods
                         if ".npz" in emb and which_method in emb]
    embedding_methods = [emb for emb in embedding_methods
                         if not ("triangle" in emb) and not
                         ("hard" in emb)]

    print "Embedding methods: {}".format(embedding_methods)
    mod = 1

    for embedding in embedding_methods:
        print "Training model %s: model %d out of %d" % \
                (embedding, mod, len(embedding_methods))
        f = np.load(embedding_path + embedding)

        x_train = np.array(f['x_train_supervised'], dtype=np.float32)
        y_train = np.array(f['y_train_supervised'])
        x_test = np.array(f['x_test_supervised'], dtype=np.float32)
        y_test = np.array(f['y_test_supervised'])

        if x_train.min() < 0 or x_test.min() < 0:
            import pdb
            pdb.set_trace()

        # Encode using triangulation
        means = x_train.mean(1)
        new_x_train = np.maximum(means[:, None] - x_train, 0)
        means = x_test.mean(1)
        new_x_test = np.maximum(means[:, None] - x_test, 0)

        # Save new encoding
        file_name = embedding_path + embedding[:-4] + "_triangle.npz"
        np.savez(file_name,
                 x_train_supervised=new_x_train,
                 y_train_supervised=y_train,
                 x_test_supervised=new_x_test,
                 y_test_supervised=y_test)

if __name__ == '__main__':

    embedding_path = "/data/lisatmp4/romerosa/feature_selection/"
    main(embedding_path)
