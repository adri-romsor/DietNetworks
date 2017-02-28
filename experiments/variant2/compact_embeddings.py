import numpy

for i in range(0,5):
    for noise in [0.25, 0.50]:
        try:
            embedding = numpy.load("all_embeddings_noise%f_fold%i.npy" % (i, noise))

            compact_embedding = embedding[::2,:] + embedding[1::2,:]
            numpy.save("all_embeddings_noise%f_compact_fold%i.npy" % (i, noise), compact_embedding)

            standardized_embedding = (compact_embedding -
                                    compact_embedding.mean(0)) / compact_embedding.std(0)
            numpy.save("all_embeddings_noise%f_standardized_fold%i.npy" % (i, noise), standardized_embedding)

            print("Embedding %i-%f processed" % (i, noise))
        except Exception, e:
            print("Embedding %i-%f NOT processed" % (i, noise))
