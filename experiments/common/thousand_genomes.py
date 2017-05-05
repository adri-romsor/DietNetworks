try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy
import os


def load_data(path="/Tmp/carriepl/Genomics_Datasets/1000_Genome_project/",
              force_pkl_recreation=False):

    dataset_file = "affy_6_biallelic_snps_maf005_thinned_aut_dataset.pkl"
    genome_file = "affy_6_biallelic_snps_maf005_thinned_aut_A.raw"
    label_file = "affy_samples.20141118.panel"
    
    if os.path.exists(path + dataset_file) and not force_pkl_recreation:
        with open(path + dataset_file, "rb") as f:
            genomic_data, label_data = pickle.load(f)
            return genomic_data, label_data
        
    print("No binary .pkl file has been found for this dataset. The data will "
          "be parsed to produce one. This will take a few minutes.")
    
    # Load the genomic data file
    with open(path + genome_file, "r") as f:
        lines = f.readlines()[1:]
    headers = [l.split()[:6] for l in lines]
    
    nb_features = len(lines[-1].split()[6:])
    genomic_data = numpy.empty((len(lines), nb_features), dtype="int8")
    for idx, line in enumerate(lines):
        if idx % 100 == 0:
            print("Parsing subject %i out of %i" % (idx, len(lines)))
        genomic_data[idx] = [int(e) for e in line.replace("NA", "0").split()[6:]]
    
    # Load the label file
    label_dict = {}
    with open(path + label_file, "r") as f:
        for line in f.readlines()[1:]:
            patient_id, ethnicity, _ = line.split()
            label_dict[patient_id] = ethnicity
            
    # Transform the label into a one-hot format
    all_labels = list(set(label_dict.values()))
    all_labels.sort()
    
    label_data = numpy.zeros((genomic_data.shape[0], len(all_labels)),
                             dtype="float32")
    for subject_idx in range(len(headers)):
        subject_id = headers[subject_idx][0]
        subject_label = label_dict[subject_id]
        label_idx = all_labels.index(subject_label)
        label_data[subject_idx, label_idx] = 1.0

    # Save the parsed data to the filesystem
    print("Saving parsed data to a binary format for faster loading in the future.")
    with open(path + dataset_file, "wb") as f:
        pickle.dump((genomic_data, label_data), f, pickle.HIGHEST_PROTOCOL)

    return genomic_data, label_data


if __name__ == '__main__':
    x = load_data(force_pkl_recreation=True)
    print("Load1 done")
    x = load_data()
    print("Load2 done")
