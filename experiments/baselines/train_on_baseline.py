import os

from featsel_supervised import execute


def main(embedding_path):
    embedding_methods = []

    for directory, _, embedding in os.walk(embedding_path):
            embedding_methods.extend([emb for emb in embedding
                                      if ".npz" in emb])

    mod = 1
    for embedding in embedding_methods:
        print "Training model %s: model %d out of %d" % \
                (embedding, mod, len(embedding_methods))
        execute(embedding, num_epochs=1000, split_valid=.15,
                lr_value=1e-3,
                save_path='/data/lisatmp4/romerosa/feature_selection/')
        mod += 1

if __name__ == '__main__':
    embedding_path = "/data/lisatmp4/romerosa/feature_selection/"
    main(embedding_path)
