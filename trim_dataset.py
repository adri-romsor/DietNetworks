import numpy


snp_to_int = {'A': 1, 'AA': 2, 'AC': 3, 'AG': 4, 'AT': 5,
              'C': 6, 'CC': 7, 'CG': 8, 'CT': 9, 'D': 10,
              'DD': 11, 'DI': 12, 'G': 13, 'GG': 14,
              'GT': 15, 'I': 16, 'II': 17, 'T': 18,
              'TT': 19, '--': 0}

int_to_snp = dict(zip(snp_to_int.values(), snp_to_int.keys()))

def replace_inplace(arr, to_replace, replace_with):
    arr += (arr == to_replace) * (replace_with - to_replace)
    

def trim_dataset():
    
    # Load the existing data
    o_data = numpy.load('/data/lisatmp4/dejoieti/ma_dataset.npy')
    
    # Step 1 : remove from the dataset all the features that have only
    # one possible value or two possible values but one of those is 'missing'
    # (0).
    print("Step 1")
    
    features_to_keep = []
    for i in range(o_data.shape[1]):
        # Let the user know how much we've done so far.
        if i % 10000 == 0:
            print(i, o_data.shape[1])
                
        uniques = numpy.unique(o_data[:,i])
        nb_uniques = len(uniques)
        
        if nb_uniques >= 3 or (nb_uniques == 2 and 0 not in uniques):
            features_to_keep.append(i)
                
    n_data = o_data[:,features_to_keep]
    
    
    # Step 2 : Replace any single-letter SNP by the double letter equivalent
    print("Step 2")
    
    numpy.place(n_data, n_data == snp_to_int['A'], snp_to_int['AA'])
    numpy.place(n_data, n_data == snp_to_int['C'], snp_to_int['CC'])
    numpy.place(n_data, n_data == snp_to_int['D'], snp_to_int['DD'])
    numpy.place(n_data, n_data == snp_to_int['G'], snp_to_int['GG'])
    numpy.place(n_data, n_data == snp_to_int['I'], snp_to_int['II'])
    numpy.place(n_data, n_data == snp_to_int['T'], snp_to_int['TT'])
    

    # Step 3 : Change the feature's representation from categories to additive
    # coding.
    print("Step 3")
    
    for i in range(n_data.shape[1]):
        # Let the user know how much we've done so far.
        if i % 10000 == 0:
            print(i, n_data.shape[1])
        
        # Determine the major allele for this feature
        allele_counts = {"A": 0, "C": 0, "D": 0, "G": 0, "I": 0, "T": 0}
        
        for snp in snp_to_int.keys():
            
            if snp != "--":

                # Determine how many times the SNP occurs in this feature
                snp_count = (n_data[:,i] == snp_to_int[snp]).sum()
                
                # Increment the proper allele counts based on the snp_count
                for allele in snp:
                    allele_counts[allele] += snp_count
                 
        max_count = max(allele_counts.values())
        major_allele = None
        for allele in allele_counts.keys():
            if allele_counts[allele] == max_count:
                major_allele = allele
                break
            
        # With the major allele known, recode the feature values
        tmp = n_data[:, i].copy()
        
        for snp in snp_to_int.keys():
            if snp != "--":
                if len(snp) == 2:
                    nb_occ = (snp[0] == major_allele) + (snp[1] == major_allele)
                elif len(snp) == 1:
                    nb_occ = int(snp == major_allele)
                numpy.place(tmp, n_data[:, i] == snp_to_int[snp], nb_occ)
        
        n_data[:, i] = tmp
    
    # Final step : Save the new dataset to disk
    print("Saving to disk")
    
    numpy.save('/data/lisatmp4/carriepl/ma_dataset_trimmed.npy', n_data)
    
    
if __name__ == "__main__":
    trim_dataset()