# TODO list

Étienne
- [X] Check Baselines PB and stats
- [X] Merge epls and master branchs
- [X] Implement EPLS in learn_model.py et learn_feat_embedding.py
- [ ] Check protein binding results, i.e. is the system learning to always say "no"? Compare the 98% from the first/last epochs. Check the number of false positives/false negatives.

Pierre Luc
- [x] Merge epls and master branchs
- [x] Reuters : check dimensionality and stats
- [x] Check if softmax has been changed to sigmoid in the new version of the code (it wasn't so I did it)
- [x] Finish refactoring dataset loader

Akram
- [x] IMDB : check dimensionality and stats
- [x] Adapt IMDB loader to fit in memory and to work with new code

Tristan
- [ ] Check Dragonn for protein 
- [ ] Check new version of the code and familiarize with it
- [ ] Start running experiments

Adriana
- [ ] Consolidate Genomics Data Commons dataset
- [ ] Check Dragonn for protein 

All
- [ ] Protein Binding (experiments)

# Pending tasks
- [ ] Train the model using partially labeled subsets (10%, 25%, 50% ...)

# Feature-Selection
Ackwnoledgement: we used scikit feature (added to the repo) for some of our baselines.
https://github.com/jundongl
