# -*- coding: utf-8 -*-
#  Author: Nicolas Boulanger-Lewandowski
# University of Montreal (2012)
# RNN-RBM deep learning tutorial
# More information at http://deeplearning.net/tutorial/rnnrbm.html

import glob
import os
import sys

import numpy
try:
    import pylab
except ImportError:
    print (
        "pylab isn't available. If you use its functionality, it will crash."
    )
    print "It can be installed with 'pip install -q Pillow'"

#from midi.utils import midiread, midiwrite
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

#Don't use a python long as this don't work on 32 bits computers.
numpy.random.seed(0xbeef)
rng = RandomStreams(seed=numpy.random.randint(1 << 30))
theano.config.warn.subtensor_merge_bug = False

#参考国内资料 http://blog.csdn.net/abcjennifer/article/details/27709915

def build_rbm(v, W, bv, bh, k):  #这是一个rbm的更新
    '''Construct a k-step Gibbs chain starting at v for an RBM.

    v : Theano vector or matrix
        If a matrix, multiple chains will be run in parallel (batch).
    W : Theano matrix
        Weight matrix of the RBM.
    bv : Theano vector
        Visible bias vector of the RBM.
    bh : Theano vector
        Hidden bias vector of the RBM.
    k : scalar or Theano scalar
        Length of the Gibbs chain.

    Return a (v_sample, cost, monitor, updates) tuple:

    v_sample : Theano vector or matrix with the same shape as `v`
        Corresponds to the generated sample(s).
    cost : Theano scalar
        Expression whose gradient with respect to W, bv, bh is the CD-k
        approximation to the log-likelihood of `v` (training example) under the
        RBM. The cost is averaged in the batch case.
    monitor: Theano scalar
        Pseudo log-likelihood (also averaged in the batch case).
    updates: dictionary of Theano variable -> Theano variable
        The `updates` object returned by scan.'''

    def gibbs_step(v):
        mean_h = T.nnet.sigmoid(T.dot(v, W) + bh)
        h = rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                         dtype=theano.config.floatX)
        mean_v = T.nnet.sigmoid(T.dot(h, W.T) + bv)
        v = rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                         dtype=theano.config.floatX)
        print "mean_v values"
        print type(mean_v)

        print mean_v.ndim

        return mean_v, v

    #updates在 lambda函数返回是 randomstream的时候有用。且randomstream的binomial的size和p都必须是symbolic的。。。
    chain, updates = theano.scan(lambda v: gibbs_step(v)[1], outputs_info=[v],
                                 n_steps=k)

    v_sample = chain[-1]

    mean_v = gibbs_step(v_sample)[0]
    monitor = T.xlogx.xlogy0(v, mean_v) + T.xlogx.xlogy0(1 - v, 1 - mean_v) #monitor(cost monitor,CD-k中用重构cross-entropy代替上面cost用来观测)
    #xlogy0 担心y0太小，会使得numpy的log报错负无穷，所以重写了这部分。
    monitor = monitor.sum() / v.shape[0]

    def free_energy(v):
        return -(v * bv).sum() - T.log(1 + T.exp(T.dot(v, W) + bh)).sum()
    cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]   # cost 是为了 对参数求导， monitor是用Cross entropy代替cost来观测

    return v_sample, cost, monitor, updates


def shared_normal(num_rows, num_cols, scale=1):
    '''Initialize a matrix shared variable with normally distributed
    elements.'''
    return theano.shared(numpy.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX))


def shared_zeros(*shape):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(numpy.zeros(shape, dtype=theano.config.floatX))


def build_rnnrbm(n_visible, n_hidden, n_hidden_recurrent): #这是多个rnn rbm的并行更新

    # rnn slu 是类似于 crf的方法，通过标记 BIO来tag 各个状态点
    # rnnrbm 是类似于 hmm的方法，通过将一整个观察序列输入到训练的model中去，然后通过 cost，也就是输入序列与 用 params（bh，bv 和W，如果一般的rbm，则W便可以判断v和vsample的相似度） 产生的序列的 相似度，如果相似度很高，
    # 则可以判断该序列是这个model产生的。


    '''Construct a symbolic RNN-RBM and initialize parameters.
    ### rmbrnn的权值是共享的 这些边在不同level（sequence的不同时刻）是共享权值滴。
    ###我们实际上要优化的参数就是上面的5个weight matrix加bv,bh,bu
    n_visible : integer
        Number of visible units.
    n_hidden : integer
        Number of hidden units of the conditional RBMs.
    n_hidden_recurrent : integer
        Number of hidden units of the RNN.

    Return a (v, v_sample, cost, monitor, params, updates_train, v_t,
    updates_generate) tuple:

    v : Theano matrix
        Symbolic variable holding an input sequence (used during training)
    v_sample : Theano matrix
        Symbolic variable holding the negative particles for CD log-likelihood
        gradient estimation (used during training)
    cost : Theano scalar
        Expression whose gradient (considering v_sample constant) corresponds
        to the LL gradient of the RNN-RBM (used during training)
    monitor : Theano scalarLL
        Frame-level pseudo-likelihood (useful for monitoring during training)
    params : tuple of Theano shared variables
        The parameters of the model to be optimized during training.
    updates_train : dictionary of Theano variable -> Theano variable
        Update object that should be passed to theano.function when compiling
        the training function.
    v_t : Theano matrix
        Symbolic variable holding a generated sequence (used during sampling)
    updates_generate : dictionary of Theano variable -> Theano variable
        Update object that should be passed to theano.function when compiling
        the generation function.'''


    W = shared_normal(n_visible, n_hidden, 0.01)
    bv = shared_zeros(n_visible)
    bh = shared_zeros(n_hidden)
    Wuh = shared_normal(n_hidden_recurrent, n_hidden, 0.0001)
    Wuv = shared_normal(n_hidden_recurrent, n_visible, 0.0001)
    Wvu = shared_normal(n_visible, n_hidden_recurrent, 0.0001)
    Wuu = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    bu = shared_zeros(n_hidden_recurrent)




    params = W, bv, bh, Wuh, Wuv, Wvu, Wuu, bu  # learned parameters as shared
                                                # variables

    v = T.matrix()  # a training sequence
    u0 = T.zeros((n_hidden_recurrent,))  # initial value for the RNN hidden
                                         # units

    # If `v_t` is given, deterministic recurrence to compute the variable
    # biases bv_t, bh_t at each time step. If `v_t` is None, same recurrence
    # but with a separate Gibbs chain at each time step to sample (generate)
    # from the RNN-RBM. The resulting sample v_t is returned in order to be
    # passed down to the sequence history.
    def recurrence(v_t, u_tm1):
            #recurrence的作用是让过去的行为状态影响

        bv_t = bv + T.dot(u_tm1, Wuv)
        # u_tm1 == u_(t-1)  u_tp1 == u_(t+1)

        bh_t = bh + T.dot(u_tm1, Wuh)
        generate = v_t is None

        if generate:
            #train 的recurrence 不需要 rbm
            v_t, _, _, updates = build_rbm(T.zeros((n_visible,)), W, bv_t,
                                           bh_t, k=25)

        u_t = T.tanh(bu + T.dot(v_t, Wvu) + T.dot(u_tm1, Wuu))
        return ([v_t, u_t], updates) if generate else [u_t, bv_t, bh_t]

    # For training, the deterministic recurrence is used to compute all the
    # {bv_t, bh_t, 1 <= t <= T} given v. Conditional RBMs can then be trained
    # in batches using those parameters.

    #u_t bv_t bh_t 是 0 < t < T 时刻的u_t，bv_t， bh_t
    (u_t, bv_t, bh_t), updates_train = theano.scan(
        lambda v_t, u_tm1, *_: recurrence(v_t, u_tm1),
        sequences=v, outputs_info=[u0, None, None], non_sequences=params)
    # scan 的updates是 储存 键值对 或者 ordered dict 其中 key 必须为 shared value
    #gibbs sampling is 15 steps

    v_sample, cost, monitor, updates_rbm = build_rbm(v, W, bv_t[:], bh_t[:],

                                                     k=15)
    #这是同时更新训练 n个时间的 rbm 。


    print "before update"
    print updates_train
    updates_train.update(updates_rbm)
    print "updates_train"
    print updates_train

    # symbolic loop for sequence generation
    (v_t, u_t), updates_generate = theano.scan(
        lambda u_tm1, *_: recurrence(None, u_tm1),
        outputs_info=[None, u0], non_sequences=params, n_steps=200)

    return (v, v_sample, cost, monitor, params, updates_train, v_t,
            updates_generate)


class RnnRbm:
    '''Simple class to train an RNN-RBM from MIDI files and to generate sample
    sequences.'''
    ## rbm的作用主要是复现 观察序列

    def __init__(
        self,
        n_hidden=150,
        n_hidden_recurrent=100,
        lr=0.001,
        n_visible=0,
        event_type=None,
        score_params_dict=None  #如果train ，则这里为None。如果这里是各个params的参数。则该模型是预测模型
    ):
        '''Constructs and compiles Theano functions for training and sequence
        generation.

        n_hidden : integer
            Number of hidden units of the conditional RBMs.
        n_hidden_recurrent : integer
            Number of hidden units of the RNN.
        lr : float
            Learning rate
        n_visible : visible state
        r : (integer, integer) tuple
            Specifies the pitch range of the piano-roll in MIDI note numbers,
            including r[0] but not r[1], such that r[1]-r[0] is the number of
            visible units of the RBM at a given time step. The default (21,
            109) corresponds to the full range of piano (88 notes).
        '''
        self.event_type = event_type
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_hidden_recurrent = n_hidden_recurrent

        #training时，先建立一个 rnnrbm模型，从u0和vt出发，利用已有的权值和偏置项来生成一个rnnrbm
        (v, v_sample, cost, monitor, params, updates_train, v_t,
            updates_generate) = build_rnnrbm(
                n_visible, #visible  midi notes
                n_hidden,   # rbm hidden
                n_hidden_recurrent    # rnn hidden
            )


        if score_params_dict != None:
            # @params.setter
            # def params(self, params_dict):
            #     params_strs = ['W', 'bv', 'bh', 'Wuh', 'Wuv', 'Wvu', 'Wuu', 'bu']
            #     params_values = [params_dict[name] for name in params_strs]
            #     self.set_params_function(params_values)
            params_strs = ['W', 'bv', 'bh', 'Wuh', 'Wuv', 'Wvu', 'Wuu', 'bu']
            [param.set_value(score_params_dict[name]) for param,name in zip(params,params_strs)]  #set each shared_value value

            self.score_function = theano.function(
            [v],
            cost,
            updates=updates_train
            )

        #先调train接口等问题。然后写score的接口

        # score function don't minus the error derivative

        # just for checking the score function didn't change the params
        self.check_params_function = theano.function(
            [],
            params
        )

        #利用自由能算出的cost的grad来更新各个 params （Wuv，Wuu，Wuh，bh等）
        gradient = T.grad(cost, params, consider_constant=[v_sample])
        updates_train.update(
            ((p, p - lr * g) for p, g in zip(params, gradient))
        )
        self.train_function = theano.function(
            [v],
            monitor,
            updates=updates_train
        )
        self.generate_function = theano.function(
            [],
            v_t,
            updates=updates_generate
        )



    def __str__(self):

        model_str = "RnnRbm instance, model name is {3} network structure is {0} visible units,{1} rnn hidden units and {2} conditional rbm hidden units".\
                    format(self.n_visible,
                           self.n_hidden_recurrent,
                           self.n_hidden,
                           self.event_type)

        print model_str,type(model_str)
        return model_str


    def train(self, dataset, batch_size=100, num_epochs=2):
        '''Train the RNN-RBM via stochastic gradient descent (SGD) using MIDI
        files converted to piano-rolls.

        files : list of strings
            List of MIDI files that will be loaded as piano-rolls for training.

        dataset : dataset 是一个model下的所有待训练序列的集合
        batch_size : integer
            Training sequences will be split into subsequences of at most this
            size before applying the SGD updates.
        num_epochs : integer
            Number of epochs (pass over the training set) performed. The user
            can safely interrupt training with Ctrl+C at any time.'''

        assert len(dataset) > 0, 'Training set is empty!' \
                               ' (did you put the right dataset?)'
        # dataset = [midiread(f, self.r,
        #                     self.dt).piano_roll.astype(theano.config.floatX)
        #            for f in files]

        #print dataset[0].shape
        #print dataset[1].shape
        # print "dataset 0 is"
        # for i in dataset[0]:
        #     print i

        try:
            for epoch in xrange(num_epochs):  #一个epoch 对所有数据训练一遍
                numpy.random.shuffle(dataset)
                costs = []

                for s, sequence in enumerate(dataset):  #输入的sequence中的element只能是一个 0，1组成的 array
                    #print "len of sequence",len(sequence)
                    for i in xrange(0, len(sequence), batch_size): # 因为 一个event的序列长度并不长，所以100的batch_size相当于一个序列一个序列训练
                        #由于 权值各个节点共享，所以，可以minibatch的训练，甚至可以online learning

                        cost = self.train_function(sequence[i:i + batch_size])
                        costs.append(cost)

                print 'Epoch %i/%i' % (epoch + 1, num_epochs),
                print numpy.mean(costs)
                sys.stdout.flush()

        except KeyboardInterrupt:
            print 'Interrupted by user.'

        #todo need to verify result in the validation set,if no improvement, stop!


    def generate(self, filename, show=True):
        '''Generate a sample sequence, plot the resulting piano-roll and save
        it as a MIDI file.

        filename : string
            A MIDI file will be created at this location.
        show : boolean
            If True, a piano-roll of the generated sequence will be shown.'''

        piano_roll = self.generate_function()
        #midiwrite(filename, piano_roll, self.r, self.dt)
        #
        if show:
            #extent = (0, self.dt * len(piano_roll)) + self.r  #压缩了上下左右四个方向的坐标。
            extent = (0, 0, 0, 0)
            pylab.figure()
            pylab.imshow(piano_roll.T, origin='lower', aspect='auto',
                         interpolation='nearest', cmap=pylab.cm.gray_r,
                         extent=extent)
            pylab.xlabel('time (s)')
            pylab.ylabel('MIDI note number')
            pylab.title('generated piano-roll')

    def score(self,seq):

        assert len(seq) > 0, 'Training set is empty!' \
                               ' (are you dumb?)'

        log_prob = self.score_function(seq)

        return log_prob


    @property
    def params(self):

        params = self.check_params_function()
        params_dic = dict(W=params[0], bv=params[1], bh=params[2], Wuh=params[3], Wuv=params[4], Wvu=params[5], Wuu=params[6], bu=params[7])

        return params_dic




    def save_params(self):
        '''
        save pararms to the local json file
        :return:
        '''
        params = self.check_params_function()
        params_dic = dict(W=params[0], bv=params[1], bh=params[2], Wuh=params[3], Wuv=params[4], Wvu=params[5], Wuu=params[6], bu=params[7])
        try:
            import json
            import time
            local_time = time.localtime()
            local_time_str = time.strftime("%Y_%m_%d_%H_%M_%S",local_time)
        except Exception,e:
            print "Exception is",e

        try:
            fp = open("./rnnrbm_" + local_time_str + ".json", "w")
            json.dump(params_dic,fp)
        except Exception,e:
            print "Exception is",e

        return "params saved successfully"







class Comparator(object):

    def __init__(self,
                 event_params_dict=None, # key is the eventtype ,and the value is the params of RnnRbm,
                                         #如果event_params_dict不为空，则fit时，是从之前训练的基础上开始训练，如果是空。则是用buildrbmrnn的
                                         #default参数来初始化并开始训练
                 event_list=None,
                 weights_dict=None,
                 senz_len = 0
                 ):

        self.event_params_dict = event_params_dict
        if weights_dict != None:
            self.weights_dict = weights_dict
        self.senz_len = senz_len
    #     RnnRbm(
    #     self,
    #     n_hidden=150,
    #     n_hidden_recurrent=100,
    #     lr=0.001,
    #     n_visible=0,
    #     event_type=None
    # )
        if self.event_params_dict != None:

            self.candidates = [(event,RnnRbm(event_type=event,n_visible=senz_len,score_params_dict=params_dict)) for event,params_dict in self.event_params_dict.items()]

        else:
            self.candidates = [(event,RnnRbm(event_type=event,n_visible=senz_len)) for event in event_list]

        temp_dict = {}
        for tup in self.candidates:
            temp_dict.update({tup[0]:tup[1]})
        self.candidates = temp_dict


    def predict(self,seq):
        temp_dict = {}

        def compute_scores(seq):
            for event,model in self.candidates.items():
                temp_dict.update({event:model.score(seq)})
                print model.params
            return temp_dict

        scores_dict = compute_scores(seq)
        print "scores_dict",scores_dict
        most_likely_event = min(scores_dict.items(),key=lambda x:abs(x[1]))
        return most_likely_event[0]

    def fit(self, seqs_dict):
        '''

        :param seqs_dict: {"dining_in_restaurant:[[one binary seq],[one binary seq]]}
        :return: params_dict: params for every model. some params are ndarray type,need to be transformed in the upper layer
        '''
        params_dict = {}
        for event,model in self.candidates.items():
            print "event",event
            print "model",model
            model.train(dataset=seqs_dict[event])
            params_dict.update({event:model.params})
        print "params dict",params_dict
        return params_dict



#def test_rnnrbm(batch_size=100, num_epochs=200):
def test_rnnrbm(batch_size=100, num_epochs=20):
    pass
    # model = RnnRbm()
    # re = os.path.join(os.path.split(os.path.dirname(__file__))[0],
    #                   'data', 'Nottingham', 'train', '*.mid')
    # print "re",re
    # print "before train params",model.params
    #
    # model.train(glob.glob(re),
    #             batch_size=batch_size, num_epochs=num_epochs)
    # return model


if __name__ == '__main__':

    pass
    # model = test_rnnrbm()
    # print "after train params",model.params
    #
    # print "before score params",model.params
    # re1 = os.path.join(os.path.split(os.path.dirname(__file__))[0],
    #                   'data', 'Nottingham', 'train', 'ashover_simple_chords*.mid')
    # print "re1", re1
    # re2 = os.path.join(os.path.split(os.path.dirname(__file__))[0],
    #                   'data', 'Nottingham', 'check_code', 'sample*.mid')
    # print "re1 differences",model.score(glob.glob(re1))  #glob return list
    # print "re2 differences", model.score(glob.glob(re2))
    # print "after score params",model.params

    #model.generate('sample1.mid')
    #model.generate('sample2.mid')

    #pylab.show()





