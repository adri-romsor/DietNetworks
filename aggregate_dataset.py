import os
import numpy as np
# import pandas as pd
import gc


def list_files(data_dir):
    def aux_fitbit(s):
        x = s.split("_")
        if len(x) < 2:
            return False
        return x[1] == "fitbit"
    # print os.listdir(data_dir)
    l = set(os.listdir(data_dir))

    data = {}
    data["23_and_me"] = list(filter(lambda l: l.endswith("23andme.txt"), l))
    l -= set(data["23_and_me"])

    data["ancestry"] = list(filter(lambda l: l.endswith("ancestry.txt"), l))
    l -= set(data["ancestry"])

    data["ftdna-illumina"] = list(filter(lambda l: l.endswith(
            "ftdna-illumina.txt"), l))
    l -= set(data["ftdna-illumina"])

    data["exome-vcf"] = list(filter(lambda l: l.endswith("exome-vcf.txt"), l))
    l -= set(data["exome-vcf"])

    data["decodeme"] = list(filter(lambda l: l.endswith("decodeme.txt"), l))
    l -= set(data["decodeme"])

    data["fitbit"] = list(filter(lambda l: aux_fitbit(l), l))
    l -= set(data["fitbit"])

    data["IYG"] = list(filter(lambda l: l.endswith("IYG.txt"), l))
    l -= set(data["IYG"])

    data["CSVs"] = list(filter(lambda l: l.endswith(".csv"), l))
    l -= set(data["CSVs"])

    data["unused"] = list(l)

    return data


def parse_illumina(filename):
    l = []
    with open(filename, "r") as input_file:
        for line in input_file:
            temp = line.split(",")
            temp = list(map(lambda l: l.strip('"\n'), temp))
            assert(len(temp) == 4)

            l.append(temp)
    return l[1:]


def parse_23_and_me(filename):
    l = []
    # with open(filename,"r",encoding="utf-8",errors='ignore') as input_file:
    with open(filename, "r") as input_file:
        for line in input_file:
            if not line.startswith("#"):
                temp = line.split("\t")
                temp = list(map(lambda l: l.strip(), temp))
                # if not (len(temp) == 4):
                #    print (temp)
                assert (len(temp) == 4)
                l.append([temp[0], temp[3]])
    return l


def parse_list(l):
    d = {}
    for i in l:
        if i[0] in d:
            assert d[i[0]] == d[1]
        else:
            d[i[0]] = i[1]
    return d


def add_dicts_validate(dict1, dict2):
    for k in dict2.keys():
        if k in dict1:
            assert dict1[k] == dict2[k]
        else:
            dict1[k] = dict2[k]


def parse_23_and_me_dict(filename, feature_values_dict):
    d = {}
    # with open(filename,"r",encoding="utf-8",errors='ignore') as input_file:
    with open(filename, "r") as input_file:
        for line in input_file:
            if not line.startswith("#"):
                temp = line.split("\t")
                temp = list(map(lambda l: l.strip(), temp))
                assert (len(temp) == 4)

                if temp[0] in d:
                    assert d[temp[0]] == feature_values_dict[temp[3]]
                else:
                    d[temp[0]] = feature_values_dict[temp[3]]
    return d


def categorize_features_for_batch(batch):
    size_batch, n_feat = batch.shape
    batch_categ = np.zeros((size_batch, 20*n_feat))
    idx_feat_start = np.arange(n_feat)*20
    slice_column_idx = np.concatenate(batch+idx_feat_start)
    slice_row_idx = np.repeat(range(size_batch), n_feat)
    batch_categ[slice_row_idx, slice_column_idx] = 1
    return batch_categ


if __name__ == "__main__":
    data_dir = "/data/lisatmp4/sylvaint/data/openSNP"
    files = list_files(data_dir)

    # print (type(files))
    # print (type(files["ftdna-illumina"]))
    # test_file = os.path.join(data_dir,files["ftdna-illumina"][0])
    # print (test_file)
    test_file = os.path.join(data_dir, files["23_and_me"][0])
    # print (parse_illumina(test_file))
    print ("Parsing 23 and me data")

    feature_values_dict = {'A': 1, 'AA': 2, 'AC': 3, 'AG': 4, 'AT': 5,
                           'C': 6, 'CC': 7, 'CG': 8, 'CT': 9, 'D': 10,
                           'DD': 11, 'DI': 12, 'G': 13, 'GG': 14,
                           'GT': 15, 'I': 16, 'II': 17, 'T': 18,
                           'TT': 19, '--': 0}

    data23andme = {}

    for index, filename in enumerate(files["23_and_me"]):
        print("Processing %i out of %i" % (index, len(files["23_and_me"])))
        input_file = os.path.join(data_dir, filename)
        user_id = filename.split("_")[0]
        try:
            temp_dict = parse_23_and_me_dict(input_file, feature_values_dict)

            if user_id in data23andme:
                add_dicts_validate(data23andme[user_id], temp_dict)
            else:
                data23andme[user_id] = temp_dict

        except:
            print("Skipping user", user_id)
        gc.collect()

    # Convert the result to a numpy array
    #####################################

    # Step 1 : create a set of all the feature ids and a dictionary mapping
    # feature ids to numbers
    feature_set = set()
    for d in data23andme.values():
        feature_set = feature_set.union(d.keys())

    feature_dict = {}
    for i, feature_id in enumerate(feature_set):
        feature_dict[feature_id] = i

    # Step 2 : allocate a numpy array large enough to contain the results
    shape = (len(data23andme.keys()), len(feature_dict.keys()))
    arr = np.zeros(shape, "int32")

    # Step 3 : go over the dict of dicts and insert the data in the allocated
    # array.
    for user_idx, user_dict in enumerate(data23andme.values()):
        if user_idx % 10 == 0:
            print user_idx
        for feature_id in user_dict.keys():

            feature_value = user_dict[feature_id]
            feature_idx = feature_dict[feature_id]

            arr[user_idx, feature_idx] = feature_value

    # Save the result to disk
    np.save("/data/lisatmp4/carriepl/ma_dataset.npy", arr)

##########################################################################
# splitting dataset
# start by filtering data by annotation availability (here label = height)


def map_to_float(s):
    if s == '':
        return -1.0
    else:
        return float(s)

height = np.genfromtxt('height.csv', dtype=None, delimiter="\t")
height[:, 1] = map(map_to_float, height[:, 1])
height = height.astype(float)  # from str to float
# height.astype(int)  # if you prefer int
# selecting the subset of user without height data
user_idx_wo_height = height[height[:, 1] == -1, 0]
user_idx_w_height = height[height[:, 1] != -1, 0]
