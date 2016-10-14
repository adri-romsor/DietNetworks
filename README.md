# TODO list

**Proof of concept:**
- [ ] Generate bigram/trigram IMDB dataset
- [ ] Train IMDB with weight-auxiliary-model reducing the number of unlabeled samples to emulate fat data (concatenating uni/bi/tri grams to increase data dimensionality).
- [ ] Train IMDB regular-AE MLP reducing the number of unlabeled samples to emulate fat data (concatenating uni/bi/tri grams to increase data dimensionality).
- [x] Train the model using partially labeled subsets (10%, 25%, 50% ...): the model does not seem to suffer from using few labeled samples.

What we would like to see: regular-AE MLP overfits whereas weight-aux-model does not.

**IMBD comparison:**
- [ ] Run experiments on https://github.com/mesnilgr/nbsvm using uni/bi/tri grams.
- [ ] Run experiments on https://github.com/libofang/DV-ngram using uni/bi/tri grams.
- [ ] Run experiments on https://github.com/mesnilgr/iclr15 using uni/bi/tri grams.

**1000 genomes:**
- [x] Wrap dataset
- [ ] Run experiments
- [ ] Run PCA/k-means baselines

# Feature-Selection
Ackwnoledgement: we used scikit feature (added to the repo) for some of our baselines.
https://github.com/jundongl
