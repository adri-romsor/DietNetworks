import numpy

for i in range(1,2):
    try:
        embedding = numpy.load("all_embeddings_fold%i.npy" % i)
        
        compact_embedding = embedding[::2,:] + embedding[1::2,:]
        numpy.save("all_embeddings_fold%i_compact.npy" % i, compact_embedding)
        
        standardized_embedding = (compact_embedding -
                                  compact_embedding.mean(0)) / compact_embedding.std(0)
        numpy.save("all_embeddings_fold%i_standardized.npy" % i, standardized_embedding)
        
        print("Embedding %i processed" % i)
    except Exception, e:
        print("Embedding %i NOT processed" % i)
