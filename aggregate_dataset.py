import os
import numpy as np
import pandas as pd


def list_files (data_dir):
    def aux_fitbit(s):
        x = s.split("_")
        if len(x) < 2: return False
        return x[1] == "fitbit"
    #print os.listdir(data_dir)
    l = set(os.listdir(data_dir))

    data = {}
    data["23_and_me"] = list(filter(lambda l: l.endswith("23andme.txt"),l))
    l -= set(data["23_and_me"])

    data["ancestry"] = list(filter(lambda l: l.endswith("ancestry.txt"),l))
    l -= set(data["ancestry"])

    data["ftdna-illumina"] = list(filter(lambda l: l.endswith("ftdna-illumina.txt"),l))
    l -= set(data["ftdna-illumina"])

    data["exome-vcf"] = list(filter(lambda l: l.endswith("exome-vcf.txt"),l))
    l -= set(data["exome-vcf"])

    data["decodeme"] = list(filter(lambda l: l.endswith("decodeme.txt"),l))
    l -= set(data["decodeme"])

    data["fitbit"] = list(filter(lambda l: aux_fitbit(l),l))
    l -= set(data["fitbit"])

    data["IYG"] = list(filter(lambda l: l.endswith("IYG.txt"),l))
    l -= set(data["IYG"])

    data["CSVs"] = list(filter(lambda l: l.endswith(".csv"),l))
    l -= set(data["CSVs"])

    data["unused"] = list(l)

    return data

def parse_illumina(filename):
    l = []
    with open(filename,"r",encoding="utf-8") as input_file:
        for line in input_file:
            temp = line.split(",")
            temp = list(map(lambda l: l.strip('"\n'),temp))
            assert(len(temp) == 4)

            l.append(temp)
    return l[1:]

def parse_23_and_me(filename):
    l = []
    with open(filename,"r",encoding="utf-8",errors='ignore') as input_file:
        for line in input_file:
            if not line.startswith("#"):
                temp = line.split("\t")
                temp = list(map(lambda l: l.strip('"\n'),temp))
                #if not (len(temp) == 4):
                #    print (temp)
                assert (len(temp) == 4)
                l.append(temp)
    return l

def parse_list (l):
    d = {}
    for i in l:
        if i[0] in d:
            assert d[i[0]] == d[3]
        else:
            d[i[0]] = i[3]
    return d


if __name__ == "__main__":
    data_dir = "/data/lisatmp4/sylvaint/data/openSNP"
    files = list_files(data_dir)

    #print (type(files))
    #print (type(files["ftdna-illumina"]))
    #test_file = os.path.join(data_dir,files["ftdna-illumina"][0])
    #print (test_file)
    test_file = os.path.join(data_dir,files["23_and_me"][0])
    #print (len(parse_23_and_me(test_file)))
    #print (parse_illumina(test_file))
    print ("Parsing 23 and me data")

    data23andme = {}

    for filename in files["23_and_me"][:2]:
        input_file = os.path.join(data_dir,filename)
        user_id = filename.split("_")[0]
        temp = parse_23_and_me(input_file)
        print (user_id)
        if user_id in data23andme:
            data23andme[user_id] += temp
        else:
            data23andme[user_id] = temp

    #for i in data23andme.keys():
    #    data23andme[i] = np.array(data23andme[i])
    #print (data23andme)
    for i in data23andme.keys():
        data23andme[i] = parse_list(data23andme[i])

    #print (data23andme)
    print ("23 and me")
    print (pd.DataFrame.from_dict(data23andme,'index'))
