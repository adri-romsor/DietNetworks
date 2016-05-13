from sklearn.cross_decomposition import PLSRegression
# from experiments.common.dorothea import load_data
from aggregate_dataset import load_data23andme_baselines

import argparse
import os
import numpy as np

def pls(dataset, n_comp, save_path):
    # Load data
    print "Loading data"
    if dataset == "opensnp":
        train_supervised, test_supervised, unsupervised = \
            load_data23andme_baselines(data_path="/Tmp/sylvaint")
    else:
        raise ValueError("Unknown dataset")

    # Save embeddings for supervised training
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print "Applying PLS"
    # Extract PCA from unsupervised data
    pls = PLSRegression(n_components = n_comp)
    pls.fit(train_supervised[0],train_supervised[1])
    # Apply PCA to supervised training data
    new_x_train_supervised = pls.transform(train_supervised[0])
    new_x_test_supervised = pls.transform(test_supervised[0])

    print "Saving embeddings"
    file_name = save_path + 'pls_' + str(n_comp) + '_embedding.npz'
    np.savez(file_name, x_train_supervised=new_x_train_supervised[:n_comp],
             y_train_supervised=train_supervised[1],
             x_test_supervised=new_x_test_supervised[:n_comp],
             y_test_supervised=test_supervised[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""PLS embedding""")

    parser.add_argument('-d',
                        type=str,
                        default="opensnp",
                        help='dataset')
    parser.add_argument('-save_path',
                        '-sp',
                        default='/data/lisatmp4/sylvaint/data/feature_selection/',
                        help='number of components for embedding')

    args = parser.parse_args()

    pls(args.d, 100,
        args.save_path)
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
