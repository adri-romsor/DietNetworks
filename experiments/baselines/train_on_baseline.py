import os

from featsel_supervised import execute


def main(embedding_path):
    embedding_methods = []

    for directory, _, embedding in os.walk(embedding_path):
            embedding_methods.extend([emb for emb in embedding
                                      if ".npz" in emb])

    mod = 1
    lr_candidates = [5*1e-5, 1e-5, 1e-4, 1e-3, 1e-2]

    #base_save_path = '/data/lisatmp4/sylvaint/feature_selection_results/'
    for lr_value in lr_candidates:
        #save_path = base_save_path + str(lr_value) + "/"
        print "learning rate: {}".format(lr_value)
        #print "save_path: {}".format(save_path)

        #if not os.path.exists(save_path):
        #    os.makedirs(save_path)

        for embedding in embedding_methods:
            print "Training model %s: model %d out of %d" % \
                    (embedding, mod, len(embedding_methods))
            execute(embedding, num_epochs=1000, split_valid=.15,
                    lr_value=lr_value,
                    save_path=embedding_path)
            mod += 1

if __name__ == '__main__':
    #embedding_path = "/data/lisatmp4/romerosa/feature_selection/"
    embedding_path = "/data/lisatmp4/sylvaint/data/feature-selection-datasets/"
    main(embedding_path)
