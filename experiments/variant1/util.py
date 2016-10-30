import numpy

import theano
import theano.tensor as tensor

from blocks.bricks.cost import Cost

from fuel.datasets import Dataset, IndexableDataset
from fuel.transformers import Transformer


class BinaryMisclassificationRate(Cost):
    """Calculates the binary misclassification rate for a mini-batch.
    """
    def __init__(self):
        super(BinaryMisclassificationRate, self).__init__()

    def apply(self, y, y_hat):
        mistakes = tensor.neq(y, tensor.gt(y_hat, 0.5))
        return mistakes.mean(dtype=theano.config.floatX)


class FeatureRepresentationToInput(Transformer):
    def __init__(self, data_stream, input_dataset, features_dataset, **kwargs):
        self.input_dataset = input_dataset
        self.features_dataset = features_dataset
        super(FeatureRepresentationToInput, self).__init__(
            data_stream=data_stream,
            produces_examples=data_stream.produces_examples,
            **kwargs)

    def transform_example(self, example):
        if 'features' in self.sources:
            example = list(example)
            index = self.sources.index('features')
            import pdb; pdb.set_trace()
        return example

    def transform_batch(self, batch):
        if 'features' in self.sources:
            batch = list(batch)
            index = self.sources.index('features')

            input_rep = self.input_dataset[batch[index][:, 0],
                                           batch[index][:, 1]][:, None]

            # Use a preallocated memory buffer to speedup the process, if
            # it exists and has the right shape.
            if (hasattr(self, 'batch_prealloc') and
                self.batch_prealloc.shape == (batch[index].shape[0],
                                              self.features_dataset.shape[0] + 1)):
                self.batch_prealloc[:, -1] = input_rep[:, 0]
            else:
                feature_rep = self.features_dataset[:,batch[index][:,1]].transpose()
                self.batch_prealloc = numpy.concatenate((feature_rep, input_rep), axis=1)
            batch[index] = self.batch_prealloc

            # feature_rep = self.features_dataset[:,batch[index][:,1]].transpose()
            # batch[index] = numpy.concatenate((feature_rep, input_rep), axis=1)

        else:
            import pdb; pdb.set_trace()
        return batch
