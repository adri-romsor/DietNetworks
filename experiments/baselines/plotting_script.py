import os

import numpy as np

import matplotlib.pyplot as plt


def main(results_path, which_method="pca", enc='triangle', n_comps=None):

    print "Starting main loop"

    best_res = np.zeros((6, len(n_comps)))

    colors = ['r', 'g', 'b', 'k', 'c', 'm', 'y']
    metrics = ["train_err", "valid_err", "test_err",
               "train_acc", "valid_acc", "test_acc"]

    for comp in range(len(n_comps)):
        comp_results = []
        for directory, _, res in os.walk(results_path):
            comp_results.extend([r for r in res if "errors" in r and
                                 which_method + "_" +
                                 str(n_comps[comp]) + "_" in r])

        if which_method == "kmeans":
            comp_results = [r for r in comp_results if enc in r]

        for res in range(len(comp_results)):
            loaded_res = np.load(results_path+comp_results[res])

            # Choose best result according to validation score
            if res == 0 or loaded_res["valid_acc"] > best_res[4, comp]:
                best_res[0, comp] = loaded_res[metrics[0]]
                best_res[1, comp] = loaded_res[metrics[1]]
                best_res[2, comp] = loaded_res[metrics[2]]
                best_res[3, comp] = loaded_res[metrics[3]]
                best_res[4, comp] = loaded_res[metrics[4]]
                best_res[5, comp] = loaded_res[metrics[5]]

    np.savez(results_path+which_method+"_to_plot_acc",
             best_err=best_res,
             n_comps=n_comps)

    for m in range(3, len(metrics)-1):
        plt.plot(n_comps, best_res[m, :], colors[m], label=metrics[m])

    plt.axis([0, n_comps[-1]+100, 0.4, 1.0])
    plt.legend()
    plt.show()

    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    results_path = \
        "/data/lisatmp4/romerosa/DietNetworks/1000_genomes/results/"

    which_method = 'kmeans'

    if which_method == 'pca':
        n_comps = [1, 2, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000, 1200,
                   1400, 1600, 2000, 2400, 2600, 2760]
    elif which_method == 'kmeans':
        n_comps = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    else:
        raise ValueError('Unknown method')

    main(results_path, which_method=which_method, enc='triangle', n_comps=n_comps)
