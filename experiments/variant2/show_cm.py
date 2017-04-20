import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from getpass import getuser

from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot(dataset,
         metric='loss',
         model_path=None,
         models=None,
         colors=None):

    cm_eth_tot = np.zeros((26, 26))
    cm_cont_tot = np.zeros((5, 5))

    for m in models:
        file_path = os.path.join(model_path, dataset, m)
        if not os.path.exists(file_path):
            raise ValueError('The path to {} does not exist'.format(file_path))

        cm_var = np.load(os.path.join(file_path, 'cm'+m[-1]+'.npz'))
        cm_eth_tot += cm_var['cm_e']
        cm_cont_tot += cm_var['cm_c']

    cm_eth_tot /= 5
    cm_cont_tot /= 5

    # Normalize
    cm_eth_tot /= cm_eth_tot.sum(0)
    cm_cont_tot /= cm_cont_tot.sum(0)

    labels = ['ACB', 'ASW', 'BEB', 'CDX', 'CEU', 'CHB', 'CHS', 'CLM', 'ESN',
              'FIN', 'GBR', 'GIH', 'GWD', 'IBS', 'ITU', 'JPT', 'KHV', 'LWK',
              'MSL', 'MXL', 'PEL', 'PJL', 'PUR', 'STU', 'TSI', 'YRI']
    labels_cont = ['EAS', 'EUR', 'AFR', 'AMR', 'SAS']

    #
    # CM ethnicities
    #

    fig = plt.figure(figsize=(9, 9))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(cm_eth_tot, cmap=plt.cm.jet, interpolation='nearest')

    width, height = cm_eth_tot.shape
    plt.xticks(range(26), labels, rotation='vertical')
    plt.yticks(range(26), labels)

    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='on',
    pad=-2) # labels along the bottom edge are on

    plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    right='off',      # ticks along the bottom edge are off
    left='off',         # ticks along the top edge are off
    labelleft='on',
    pad=-2) # labels along the bottom edge are off

    plt.xlabel('label', fontsize=18)
    plt.ylabel('prediction', fontsize=18)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig.colorbar()
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(res, ticks=[0, 0.5, 1], cax=cax)
    plt.show()
    fig.savefig('cm_eth26.png', dpi=fig.dpi)


    #
    # CM continents
    #
    fig = plt.figure(figsize=(9, 9))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(cm_cont_tot, cmap=plt.cm.jet, interpolation='nearest')

    width, height = cm_eth_tot.shape
    plt.xticks(range(5), labels_cont, rotation='vertical')
    plt.yticks(range(5), labels_cont)

    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='on',
    pad=-2) # labels along the bottom edge are on

    plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    right='off',      # ticks along the bottom edge are off
    left='off',         # ticks along the top edge are off
    labelleft='on',
    pad=-2) # labels along the bottom edge are off

    plt.xlabel('label', fontsize=18)
    plt.ylabel('prediction', fontsize=18)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig.colorbar()
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(res, ticks=[0, 0.5, 1], cax=cax)
    plt.show()
    fig.savefig('cm_cont5.png', dpi=fig.dpi)







def main():
    parser = argparse.ArgumentParser(description='Conf. mat.')
    parser.add_argument('-dataset',
                        default='1000_genomes',
                        help='Dataset')
    parser.add_argument('-metric',
                        default='loss',
                        help='Metric (acc or loss)')
    parser.add_argument('-path',
                        default='/data/lisatmp4/romerosa/DietNetworks/',
                        help='Path to errors file')
    parser.add_argument('-models',
                        type=list,
                        default=[
                        'dietnet_histo_new2histo3x26fold0_our_model1.0_lr-3e-05_anneal-0.999_eni-0.02_dni-0.02_accuracy_Ri20.0_hu-100_tenc-100-100_tdec-100-100_hs-100_fold0',
                        'dietnet_histo_new2histo3x26fold1_our_model1.0_lr-3e-05_anneal-0.999_eni-0.02_dni-0.02_accuracy_Ri20.0_hu-100_tenc-100-100_tdec-100-100_hs-100_fold1',
                        'dietnet_histo_new2histo3x26fold2_our_model1.0_lr-3e-05_anneal-0.999_eni-0.02_dni-0.02_accuracy_Ri20.0_hu-100_tenc-100-100_tdec-100-100_hs-100_fold2',
                        'dietnet_histo_new2histo3x26fold3_our_model1.0_lr-3e-05_anneal-0.999_eni-0.02_dni-0.02_accuracy_Ri20.0_hu-100_tenc-100-100_tdec-100-100_hs-100_fold3',
                        'dietnet_histo_new2histo3x26fold4_our_model1.0_lr-3e-05_anneal-0.999_eni-0.02_dni-0.02_accuracy_Ri20.0_hu-100_tenc-100-100_tdec-100-100_hs-100_fold4'
                        ],
                        help='List of model names.')
    parser.add_argument('-colors',
                        type=list,
                        default=['r', 'g', 'b', 'k', 'y'],
                        help='Colors to plot curves.')
    args = parser.parse_args()

    plot(args.dataset, args.metric, args.path, args.models, args.colors)

if __name__ == "__main__":
    main()


# ['CHB', 'JPT', 'CHS', 'CDX', 'KHV', 'CEU', 'TSI', 'FIN', 'GBR', 'IBS', 'YRI', 'LWK', 'GWD', 'MSL', 'ESN', 'ASW', 'ACB', 'MXL', 'PUR', 'CLM', 'PEL', 'GIH', 'PJL', 'BEB', 'STU', 'ITU']
# ['ACB', 'ASW', 'BEB', 'CDX', 'CEU', 'CHB', 'CHS', 'CLM', 'ESN', 'FIN', 'GBR', 'GIH', 'GWD', 'IBS', 'ITU', 'JPT', 'KHV', 'LWK', 'MSL', 'MXL', 'PEL', 'PJL', 'PUR', 'STU', 'TSI', 'YRI']
