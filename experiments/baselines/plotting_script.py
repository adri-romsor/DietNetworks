import os

import numpy as np

import matplotlib.pyplot as plt


def main(results_path, which_method="pca", n_comps=None):

    print "Starting main loop"

    best_res = np.zeros((3, len(n_comps)))

    for comp in range(len(n_comps)):
        comp_results = []
        for directory, _, res in os.walk(results_path):
            comp_results.extend([r for r in res if "errors" in r and
                                 which_method + "_" +
                                 str(n_comps[comp]) + "_" in r])

        for res in range(len(comp_results)):
            loaded_res = np.load(results_path+comp_results[res])

            # Choose best result according to validation score
            if res == 0 or loaded_res["valid_err"] < best_res[1, comp]:
                best_res[0, comp] = loaded_res["train_err"]
                best_res[1, comp] = loaded_res["valid_err"]
                best_res[2, comp] = loaded_res["test_err"]

    np.savez(results_path+which_method+"_to_plot",
             best_err=best_res,
             n_comps=n_comps)

    # plt.plot(n_comps, best_res.T)
    # plt.show()


if __name__ == '__main__':

    results_path = \
        "/data/lisatmp4/romerosa/feature_selection/results/"

    n_comps = [1, 2, 5, 10, 20, 50,
               100, 200, 400, 600, 800,
               1000, 1200, 1400, 1600, 2000]

    main(results_path, "kmeans", n_comps)
