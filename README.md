## DietNetworks: Thin parameters for Fat Genomics

This repo contains the code to reproduce the experiments of the paper [DietNetworks: Thin parameters for fat genomics](https://arxiv.org/abs/1611.09340).

# Experiments

**Diet Networks with per class histograms (fold 0):**

THEANO_FLAGS='device=gpu' python learn_model.py --embedding_source=/path/to/per_class_histograms/fold0.npy --which_fold=0 -exp_name='dietnets_' -embedding_input='histo3x26' 
