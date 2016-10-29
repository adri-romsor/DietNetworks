import os
import numpy as np


def load_data(
    path='/data/lisatmp4/dejoieti/data/Thrombin',
    shuffle=False,
    seed=0
):
    data_path = os.path.join(path, 'data.npy')
    labels_path = os.path.join(path, 'label.npy')

    data = (np.load(data_path)).astype('float32')
    labels = (np.load(labels_path)).astype('int32')

    if shuffle:
        np.random.seed(seed)
        idx_shuffle = np.random.permutation(data.shape[0])
        data = data[idx_shuffle]
        labels = labels[idx_shuffle]

    return data, labels


def data_to_numpy(path='/data/lisatmp4/dejoieti/data/Thrombin/'):

    file_path = os.path.join(path, 'thrombin.data')

    f = open(file_path)

    data = []
    labels = []
    for line in f:
        line = line.split(',')
        if line[0] == 'I':
            labels.append(0)
        elif line[0] == 'A':
            labels.append(1)
        else:
            print 'Warning, no label for this line'

        line = line[1:]
        line = map(int, line)

        data.append(line)

    data = np.array(data)
    labels = np.array(labels)

    assert data.shape[0] == labels.shape[0]

    np.save(os.path.join(path, 'data.npy'), data)
    np.save(os.path.join(path, 'label.npy'), labels)

if __name__ == '__main__':
    data_to_numpy()
