import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from getpass import getuser


def plot(dataset,
         metric='loss',
         model_path=None,
         model=None,
         colors=None):

    file_path = os.path.join(model_path, dataset, model)
    if not os.path.exists(file_path):
        raise ValueError('The path to {} does not exist'.format(file_path))

    cm_var = np.load(os.path.join(file_path, 'cm.npz'))
    cm = cm_var['arr_0']

    labels = ['ACB', 'ASW', 'BEB', 'CDX', 'CEU', 'CHB', 'CHS', 'CLM', 'ESN',
              'FIN', 'GBR', 'GIH', 'GWD', 'IBS', 'ITU', 'JPT', 'KHV', 'LWK',
              'MSL', 'MXL', 'PEL', 'PJL', 'PUR', 'STU', 'TSI', 'YRI']

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(cm, cmap=plt.cm.jet, interpolation='nearest')

    width, height = cm.shape
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(int(cm[x][y])), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    plt.xticks(range(26), labels, rotation='vertical')
    plt.yticks(range(26), labels)

    # plt.savefig('confusion_matrix.png', format='png')

    plt.xlabel('label')
    plt.ylabel('prediction')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='plot errors')
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
                        type=str,
                        default='our_model1.0_sup_good',
                        help='List of model names.')
    parser.add_argument('-colors',
                        type=list,
                        default=['r', 'g', 'b', 'k'],
                        help='Colors to plot curves.')
    args = parser.parse_args()

    plot(args.dataset, args.metric, args.path, args.models, args.colors)

if __name__ == "__main__":
    main()


# ['CHB', 'JPT', 'CHS', 'CDX', 'KHV', 'CEU', 'TSI', 'FIN', 'GBR', 'IBS', 'YRI', 'LWK', 'GWD', 'MSL', 'ESN', 'ASW', 'ACB', 'MXL', 'PUR', 'CLM', 'PEL', 'GIH', 'PJL', 'BEB', 'STU', 'ITU']
# ['ACB', 'ASW', 'BEB', 'CDX', 'CEU', 'CHB', 'CHS', 'CLM', 'ESN', 'FIN', 'GBR', 'GIH', 'GWD', 'IBS', 'ITU', 'JPT', 'KHV', 'LWK', 'MSL', 'MXL', 'PEL', 'PJL', 'PUR', 'STU', 'TSI', 'YRI']
