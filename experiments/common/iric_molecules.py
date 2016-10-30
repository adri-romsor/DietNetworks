import csv
import numpy

class IricMoleculesDataset(object):

    def __init__(self, folder="/data/lisa/data/iric_molecule_fingerprint/",
                 start=0, end=None):

        assert start == 0
        assert end is None or end > start

        self.folder = folder
        self.start = start
        self.end = end

    def load_data(self, which_source="all"):

        assert which_source in ["affinity", "fingerprint", "all"]

        # Read data file stored as Tab-separated values
        filename = self.folder + "data.txt"
        reader = csv.reader(open(filename, 'rb'), delimiter="\t")

        # Discard the first line of the file
        reader.next()

        # Parse the following lines
        ids = []
        affinities = []
        fingerprints = []
        for line in reader:
            ids.append(int(line[0]))
            affinities.append([float(e) for e in line[2:22]])
            fingerprints.append([int(e) for e in line[22:]])

        ids = numpy.array(ids)
        affinities = numpy.array(affinities).astype("float32")
        fingerprints = numpy.array(fingerprints).astype("float32")

        # Scale the affinities to the [-1, 1] range
        print("IRIC Molecule affinities scaled from [%s, %s] to [-1.0, 1.0]" %
              (affinities.min(), affinities.max()))
        affinities = (affinities - affinities.min() /
                      (affinities.max() - affinities.min()))

        # Sort the data according to molecule id
        indices = ids.argsort()
        ids = ids[indices]
        affinities = affinities[indices]
        fingerprints = fingerprints[indices]

        # Crop the data
        if self.end is None:
            ids = ids[self.start:]
            affinities = affinities[self.start:]
            fingerprints = fingerprints[self.start:]
        else:
            ids = ids[self.start:self.end]
            affinities = affinities[self.start:self.end]
            fingerprints = fingerprints[self.start:self.end]

        # Return only the requested datasources
        if which_source == "affinity":
            return affinities
        elif which_source == "fingerprint":
            return fingerprints
        else:
            return numpy.hstack((affinities, fingerprints))
