import theano
from theano import tensor

import logging
from argparse import ArgumentParser

from blocks.algorithms import GradientDescent, Scale, RMSProp
from blocks.bricks import MLP, Tanh, Softmax, Logistic, Rectifier
from blocks.bricks.cost import (CategoricalCrossEntropy, MisclassificationRate,
                                BinaryCrossEntropy)
from blocks.initialization import IsotropicGaussian, Constant
from fuel.streams import DataStream
from fuel.transformers import Flatten
from fuel.datasets import IterableDataset, MNIST, IndexableDataset
from fuel.schemes import SequentialScheme
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop
from blocks.roles import WEIGHT

try:
    from blocks_extras.extensions.plot import Plot
    BLOCKS_EXTRAS_AVAILABLE = True
except:
    BLOCKS_EXTRAS_AVAILABLE = False

from util import (BinaryMisclassificationRate,
                  FeatureRepresentationToInput)

# Load the data and format it
from feature_selection.experiments.common import dorothea
train_x = dorothea.load_data("train", "standard", False, "numpy")[0]
valid_x = dorothea.load_data("valid", "standard", False, "numpy")[0]
train_set = dorothea.load_data("train", "feature_representation", True, "fuel")
valid_set = dorothea.load_data("valid", "feature_representation", True, "fuel")

nb_examples, nb_features = train_x.shape


def main(save_to, num_epochs):
    mlp = MLP([Rectifier(), Rectifier(), Logistic()],
              [nb_examples + 1, 4000, 4000, 1],
              weights_init=IsotropicGaussian(0.01),
              biases_init=Constant(0))
    mlp.initialize()

    x = tensor.matrix('features')          # shape : b * (1 + d)
    y = tensor.vector('targets')          # shape : b

    preds = mlp.apply(x)

    cost = BinaryCrossEntropy().apply(y[:, None], preds)

    # Compute various error rates for monitoring
    error_rate = BinaryMisclassificationRate().apply(y[:, None], preds)
    error_rate.name = 'individual_error_rate'

    # g_preds = preds.reshape([nb_features, -1])
    # g_y = y.reshape([nb_features, -1])
    # averaging_error = BinaryMisclassificationRate().apply(g_y[0,:], g_preds.mean(axis=1))
    # averaging_error.name = 'averaging_error_rate'
    # voting_error = BinaryMisclassificationRate().apply(g_y[0,:],
    #                                                   tensor.gt(g_preds, 0.5).mean(axis=1))
    # voting_error.name = 'voting_error_rate'

    averaging_error = BinaryMisclassificationRate().apply(y[0], preds.mean())
    averaging_error.name = 'averaging_error_rate'
    voting_error = BinaryMisclassificationRate().apply(
        y[0], tensor.gt(preds, 0.5).mean())
    voting_error.name = 'voting_error_rate'

    cg = ComputationGraph([cost])
    # W1, W2, W3 = VariableFilter(roles=[WEIGHT])(cg.variables)
    # cost = (cost + .00005 * (W1 ** 2).sum() +
    #               .00005 * (W2 ** 2).sum() +
    #               .00005 * (W3 ** 2).sum())
    cost.name = 'final_cost'

    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=RMSProp(learning_rate=0.01))

    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs),
                  DataStreamMonitoring(
                      [cost, error_rate, averaging_error, voting_error],
                      FeatureRepresentationToInput(
                          DataStream.default_stream(
                              valid_set,
                              iteration_scheme=SequentialScheme(
                                  valid_set.num_examples, 100000)),
                          input_dataset=valid_x,
                          features_dataset=train_x),
                      prefix="test"),
                  TrainingDataMonitoring(
                      [cost, error_rate, averaging_error, voting_error,
                       aggregation.mean(algorithm.total_gradient_norm)],
                      prefix="train",
                      after_epoch=True),
                  Checkpoint(save_to),
                  Printing()]

    main_loop = MainLoop(
        algorithm,
        FeatureRepresentationToInput(
            DataStream.default_stream(
                train_set,
                iteration_scheme=SequentialScheme(
                    train_set.num_examples, 100000)),
            input_dataset=train_x,
            features_dataset=train_x),
        model=Model(cost),
        extensions=extensions)

    main_loop.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("Feature selection on the Dorothea dataset")
    parser.add_argument("--num-epochs", type=int, default=1000,
                        help="Number of training epochs to do.")
    parser.add_argument("save_to", default="mnist.pkl", nargs="?",
                        help=("Destination to save the state of the training "
                              "process."))
    args = parser.parse_args()
    main(args.save_to, args.num_epochs)


# Build the model
def construct_model(input_dim, output_dim):
    # Construct the model
    r = tensor.fmatrix('r')
    x = tensor.fmatrix('x')
    y = tensor.ivector('y')

    nx = x.shape[0]
    nj = x.shape[1]  # also is r.shape[0]
    nr = r.shape[1]

    # r is nj x nr
    # x is nx x nj
    # y is nx x 1

    # r_rep is nx x nj x nr
    r_rep = r[None, :, :].repeat(axis=0, repeats=nx)
    # x3 is nx x nj x 1
    x3 = x[:, :, None]

    # concat is nx x nj x (nr + 1)
    concat = tensor.concatenate([r_rep, x3], axis=2)
    mlp_input = concat.reshape((nx * nj, nr + 1))

    # input_dim must be nr
    mlp = MLP(activations=activation_functions,
              dims=[input_dim+1] + hidden_dims + [output_dim])

    activations = mlp.apply(mlp_input)

    act_sh = activations.reshape((nx, nj, output_dim))
    final = act_sh.mean(axis=1)

    cost = Softmax().categorical_cross_entropy(y, final).mean()

    pred = final.argmax(axis=1)
    error_rate = tensor.neq(y, pred).mean()

    # Initialize parameters
    for brick in [mlp]:
        brick.weights_init = IsotropicGaussian(0.01)
        brick.biases_init = Constant(0.001)
        brick.initialize()

    # apply noise
    cg = ComputationGraph([cost, error_rate])
    noise_vars = VariableFilter(roles=[WEIGHT])(cg)
    apply_noise(cg, noise_vars, noise_std)
    [cost_reg, error_rate_reg] = cg.outputs

    return cost_reg, error_rate_reg, cost, error_rate
