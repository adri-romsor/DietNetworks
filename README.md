# TODO list

**Proof of concept:**
- [ ] Generate bigram/trigram IMDB dataset.
- [ ] Train IMDB with weight-auxiliary-model concatenating uni/bi/tri grams to increase data dimensionality.
- [ ] Train IMDB regular-AE MLP concatenating uni/bi/tri grams to increase data dimensionality (2 versions supervised & semi-supervised).
- [x] Train the model using partially labeled subsets (10%, 25%, 50% ...): the model does not seem to suffer from using few labeled samples.

What we would like to see: regular-AE MLP overfits whereas weight-aux-model does not.

**TODO:**
- [x] 5-fold cross-validation code
- [ ] Reduce number of free parameters on our model
- [ ] Change N values coming from N subjects (feature representation input) and use some stat information instead (mean, var, min, max...) to reduce the number of parameters of the auxiliary network.
- [ ] Add hierarchical softmax to take into account continental ethnicities
- [ ] Literature review from ML perspective, what has been done so far to address this problem?
- [ ] Add skip connections?
- [ ] BN
- [ ] L2 reg

**IMBD comparison:**
- [x] Run experiments on https://github.com/mesnilgr/nbsvm using uni/bi/tri grams.
- [x] Run experiments on https://github.com/libofang/DV-ngram using uni/bi/tri grams (results non-reproducible).
- [ ] Run experiments on https://github.com/mesnilgr/iclr15 using uni/bi/tri grams.

**1000 genomes:**
- [x] Wrap dataset
- [ ] Run experiments with different number of labeled samples
- [x] Run PCA/k-means baselines (ongoing)

# Results

**Accoracy for partially labeled subsets on IMDB:**

The results with NBSVM were obtained with the following repo : https://github.com/carriepl/nbsvm forked from Grégoire Mesnil's repo (https://github.com/mesnilgr/nbsvm). The only modification is the addition of code in nbsvm.py that supports using only a fraction of the training data and a new batch script 'less_labels.sh' that runs trains the model on various fractions of the training set using either unigram inputs, unigram+bigram inputs or unigram+bigram+trigram inputs.

Partial subset|Our model uni|nbsvm uni|nbsvm uni+bi|nbsvm uni+bi+tri|
--------------|-------------|---------|------------|----------------|
|         100%|        89.0%|   88.61%|      91.56%|          91.87%|
|          50%|        87.0%|   88.34%|      91.00%|          91.23%|
|          25%|        -----|   87.65%|      90.14%|          90.48%|
|          10%|        82.1%|   86.68%|      88.93%|          88.98%|
|           5%|        -----|   85.32%|      87.42%|          87.64%|
|           1%|        72.8%|   82.42%|      83.90%|          84.38%|

**SOTA methods on IMDB:**



**100 Genomes:**

|Model|Acc.train|Acc.val|Acc.test| Params | # free params |
|-----|---------|-------|--------|--------|---------------|
|Ours sup |100.%|90.63%|90%| 100(hu) 100(tenc) 100(tdec) 100-26(hsup)| |
|Ours sup |100.%|89.69%|88.91%| 50(hu) 100(tenc) 100(tdec) 100-26(hsup)| |
|Ours sup |99.95%|83.44%|81.56%| 10-50-100(hu) 100(tenc) 100(tdec) 100-26(hsup)| |
|Basic sup|96.63%|56.53%|62.03% | | |
|Basic sup+unsup|99.22%|68.44%|66.41% | | |
|PCA +  MLP|99.5%|86.67%|83.77%|||
|K-means soft|69.13%|64.93%|66.67%|||


*Some results on 1000 genomes*

![image](./images/cm.png)

# Feature-Selection
Ackwnoledgement: we used scikit feature (added to the repo) for some of our baselines.
https://github.com/jundongl
