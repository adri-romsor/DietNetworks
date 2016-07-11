import numpy as np
import theano
import theano.tensor as T
import theano.typed_list as tl

from lasagne.layers import Layer


def tensor_fun_EPLS(H, act, N, nb_activation, debug=False):
    nb_sample, nb_cluster = H.shape

    if debug:
        theano.config.compute_test_value = 'warn'
        nb_sample.tag.test_value = 10
        nb_cluster.tag.test_value = 100

        nb_sample_test = 10
        nb_cluster_test = 100
        H.tag.test_value = np.array(np.random.rand(
            nb_sample_test, nb_cluster_test), dtype='float32')
        act.tag.test_value = np.zeros(nb_cluster_test, dtype='float32')

    H = (H - T.min(H))/(T.max(H) - T.min(H))

    def update_act_target(i, target, a, H):
        activ = T.argsort(H[i]-act)[-nb_activation:]
        a = T.inc_subtensor(a[activ],
                            T.cast(nb_cluster, 'float32')/(N*nb_activation))
        target = T.set_subtensor(target[i, activ], 1)
        return target, a

    results, updates = theano.scan(
            fn=update_act_target,
            outputs_info=[T.zeros((nb_sample, nb_cluster)), act],
            sequences=T.arange(nb_sample),
            non_sequences=H)

    target = results[0][-1]
    activation = results[1][-1]
    # def weighted_H(j, H):
    #     Hj = T.copy(H[:, j])
    #
    #     def aux(i):
    #         if T.eq(H[i, j], 0):
    #             T.set_subtensor(Hj[i], -1)
    #     theano.scan(
    #         fn=aux,
    #         outputs_info=None,
    #         sequences=T.arrange(nb_sample))
    #     sorted_idx = T.argsort(Hj)
    #
    #     def aux2(i, weight):
    #         if Hj[i] == 0:
    #             return weight
    #         else:
    #             T.set_subtensor(H[i, j], weight)
    #             return weight/2
    #     theano.scan(
    #         fn=aux2,
    #         sequences=sorted_idx
    #     )

    return target, activation


def EPLS(N, act, debug=False, nb_activation=1):
    H = T.fmatrix()

    target, new_act = tensor_fun_EPLS(H, act, N, debug=debug,
                                      nb_activation=nb_activation)

    fun = theano.function([H], [target], updates=[(act, new_act)])
    return fun


def transposed_EPLS(N, debug=False, nb_activation=1):
    H = T.fmatrix()
    act = T.fvector()

    transposed_target, new_act = tensor_fun_EPLS(
        H.T, act.T, N, nb_activation=nb_activation, debug=debug)

    trans_fun = theano.function(
        [H, act], [transposed_target.T], updates=[(act, new_act.T)])
    return trans_fun


def test_EPLS():
    f = EPLS(50, debug=False)
    nb_sample = 50
    nb_cluster = 10
    H = np.random.rand(nb_sample, nb_cluster)
    H = np.array(H, dtype='float32')
    act = np.zeros(nb_cluster, dtype='float32')
    target, act = f(H, act)

    print target.sum(axis=0)


def test_EPLS_init():
    f = EPLS(50, debug=True, nb_activation=2)
    nb_sample = 50
    nb_cluster = 10
    H = np.zeros((nb_sample, nb_cluster))
    H = np.array(H, dtype='float32')
    act = np.zeros(nb_cluster, dtype='float32')

    target, act = f(H, act)

    print target
    print target.sum(axis=0)


def test2_EPLS():
    f = EPLS(50, nb_activation=15, debug=False)
    nb_sample = 10
    nb_cluster = 100
    H = np.random.rand(nb_sample, nb_cluster)
    H = np.array(H, dtype='float32')
    act = np.zeros(nb_cluster, dtype='float32')
    target, act = f(H, act)

    print target
    print target.sum(axis=0)
    print target.sum(axis=1)


def test_transposed_EPLS():
    nb_sample = 4
    nb_cluster = 12
    H = np.random.rand(nb_sample, nb_cluster)
    H = np.array(H, dtype='float32')
    act = np.zeros(nb_sample, dtype='float32')

    f = transposed_EPLS(20)
    target, act = f(H, act)

    print target.sum(axis=1)
    print target


class EPLSLayer(Layer):
    def __init__(self, incoming, N=None, nb_activation=1, **kwargs):
        super(EPLSLayer, self).__init__(incoming, **kwargs)

        self.num_samples = self.input_shape[0]
        self.num_inputs = self.input_shape[1]

        # parameter
        init_W = np.zeros((self.num_inputs, self.num_inputs), dtype='float32')
        self.W = self.add_param(init_W, (self.num_inputs, self.num_inputs),
                                name='W')

        if not N:
            N = max(1, 5*self.num_samples/(self.num_inputs*nb_activation))
        self.epls_fun = EPLS(N, nb_activation=nb_activation)

    def get_output_for(self, input, **kwargs):
        activation = np.zeros(self.num_inputs, dtype='float32')

        H = T.dot(input, self.W)
        target, act = self.epls_fun(self.H, activation)

        return H, target

if __name__ == '__main__':
    # test_EPLS()
    # test2_EPLS()
    test_EPLS_init()
    # test_transposed_EPLS()
