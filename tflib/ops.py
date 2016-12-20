import tensorflow as tf
import time
import numpy as np
import theano
theano.config.floatX='float32'
import lasagne
import tflib
import numpy
from tensorflow.python.ops import array_ops

'''
Wrapper function to perform different initialization techniques
'''
def initializer(
    name,
    shape,
    val=0,
    gain='linear',
    std=0.01,
    mean=0.0,
    range=0.01,
    alpha=0.01
    ):
    if gain in ['linear','sigmoid','tanh']:
        gain = 1.0
    elif gain=='leakyrelu':
        gain = np.sqrt(2/(1+alpha**2))
    elif gain=='relu':
        gain = np.sqrt(2)
    else:
        raise NotImplementedError

    if name=='Constant':
        return lasagne.init.Constant(val).sample(shape)
    elif name=='Normal':
        return lasagne.init.Normal(std,mean).sample(shape)
    elif name=='Uniform':
        return lasagne.init.Uniform(range=range,std=std,mean=mean).sample(shape)
    elif name=='GlorotNormal':
        return lasagne.init.GlorotNormal(gain=gain).sample(shape)
    elif name=='GlorotUniform':
        return lasagne.init.GlorotUniform(gain=gain).sample(shape)
    elif name=='HeNormal':
        return lasagne.init.HeNormal(gain=gain).sample(shape)
    elif name=='HeUniform':
        return lasagne.init.HeUniform(gain=gain).sample(shape)
    elif name=='Orthogonal':
        return lasagne.init.Orthogonal(gain=gain).sample(shape)
    else:
        return lasagne.init.GlorotUniform(gain=gain).sample(shape)


'''
Performs a linear / dense / affine transformation on the input data
Optionally, weight normalization can be performed
'''
def Linear(
    name,
    inputs,
    input_dim,
    output_dim,
    activation='linear',
    bias=True,
    init=None,
    weightnorm=False,
    **kwargs
    ):
    with tf.name_scope(name) as scope:
        weight_values = initializer(init,(input_dim,output_dim),gain=activation, **kwargs)

        weight = tflib.param(
            name + '.W',
            weight_values
        )

        batch_size = None

        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(weight_values), axis=0))
            # nort.m_values = np.linalg.norm(weight_values, axis=0)

            target_norms = tflib.param(
                name + '.g',
                norm_values
            )

            with tf.name_scope('weightnorm') as scope:
                norms = tf.sqrt(tf.reduce_sum(tf.square(weight), reduction_indices=[0]))
                weight = weight * (target_norms / norms)

        if inputs.get_shape().ndims == 2:
            result = tf.matmul(inputs, weight)
        else:
            reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
            result = tf.matmul(reshaped_inputs, weight)
            result = tf.reshape(result, tf.pack(tf.unpack(tf.shape(inputs))[:-1] + [output_dim]))

        if bias:
            b = tflib.param(
                name + '.b',
                numpy.zeros((output_dim,), dtype='float32')
            )

            result = tf.nn.bias_add(result,b)

        return result

'''
Performs 2D convolution in the NCHW data format
Weight normalization / batch normalization can be performed optionally
'''
def conv2d(
    name,
    input,
    kernel,
    stride,
    depth,
    num_filters,
    init = 'GlorotUniform',
    pad = 'SAME',
    bias = True,
    weightnorm = False,
    batchnorm = False,
    **kwargs
    ):
    with tf.name_scope(name) as scope:
        filter_values = initializer(init,(kernel,kernel,depth,num_filters),gain='relu',**kwargs)
        filters = tflib.param(
            name+'.W',
            filter_values
        )

        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0,1,2)))
            target_norms = tflib.param(
                name + '.g',
                norm_values
            )
            with tf.name_scope('weightnorm') as scope:
                norms = tf.sqrt(tf.reduce_sum(tf.square(filters), reduction_indices=[0,1,2]))
                filters = filters * (target_norms / norms)

        out = tf.nn.conv2d(input, filters, strides=[1, 1, stride, stride], padding=pad, data_format='NCHW')

        if bias:
            b = tflib.param(
                name+'.b',
                np.zeros(num_filters,dtype=np.float32)
            )

            out = tf.nn.bias_add(out,b,data_format='NCHW')

        if batchnorm:
            out = tf.contrib.layers.batch_norm(out,scope=scope,data_format='NCHW')

        return out

'''
Performs max pooling operation with kernel kxk and stride (s,s) on input with NCHW data foramt
'''
def max_pool(
    name,
    l_input,
    k,
    s
    ):
    if type(k)==int:
        k1=k
        k2=k
    else:
        k1 = k[0]
        k2 = k[1]
    if type(s)==int:
        s1=s
        s2=s
    else:
        s1 = s[0]
        s2 = s[1]
    return tf.nn.max_pool(l_input, ksize=[1, 1, k1, k2], strides=[1, 1, s1, s2],
                          padding='SAME', name=name, data_format='NCHW')

'''
Performs local response normalization (ref. Alexnet)
'''
def norm(
    name,
    l_input,
    lsize=4
    ):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

class GRUCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, name, n_in, n_hid):
        self._n_in = n_in
        self._n_hid = n_hid
        self._name = name

    @property
    def state_size(self):
        return self._n_hid

    @property
    def output_size(self):
        return self._n_hid

    def __call__(self, inputs, state, scope=None):
        gates = tf.nn.sigmoid(
            tflib.ops.Linear(
                self._name+'.Gates',
                tf.concat(1, [inputs, state]),
                self._n_in + self._n_hid,
                2 * self._n_hid
            )
        )

        update, reset = tf.split(1, 2, gates)
        scaled_state = reset * state

        candidate = tf.tanh(
            tflib.ops.Linear(
                self._name+'.Candidate',
                tf.concat(1, [inputs, scaled_state]),
                self._n_in + self._n_hid,
                self._n_hid
            )
        )

        output = (update * candidate) + ((1 - update) * state)

        return output, output

def GRU(
    name,
    inputs,
    n_in,
    n_hid
    ):
    h0 = tflib.param(name+'.h0', np.zeros(n_hid, dtype='float32'))
    batch_size = tf.shape(inputs)[0]
    h0 = tf.reshape(tf.tile(h0, tf.pack([batch_size])), tf.pack([batch_size, n_hid]))
    return tf.nn.dynamic_rnn(GRUCell(name, n_in, n_hid), inputs, initial_state=h0, swap_memory=True)[0]

class LSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, name, n_in, n_hid, forget_bias=1.0):
        self._n_in = n_in
        self._n_hid = n_hid
        self._name = name
        self._forget_bias = forget_bias

    @property
    def state_size(self):
        return self._n_hid

    @property
    def output_size(self):
        return self._n_hid

    def __call__(self, inputs, state, scope=None):
        c_tm1, h_tm1 = tf.split(1,2,state)
        gates = tflib.ops.Linear(
                self._name+'.Gates',
                tf.concat(1, [inputs, h_tm1]),
                self._n_in + self._n_hid,
                4 * self._n_hid,
                activation='sigmoid'
                )

        i_t,f_t,o_t,g_t = tf.split(1, 4, gates)

        c_t = tf.nn.sigmoid(f_t+self._forget_bias)*c_tm1 + tf.nn.sigmoid(i_t)*tf.tanh(g_t)
        h_t = tf.nn.sigmoid(o_t)*tf.tanh(c_t)

        new_state = tf.concat(1, [c_t,h_t])

        return h_t,new_state

def LSTM(
    name,
    inputs,
    n_in,
    n_hid,
    h0
    ):
    return tf.nn.dynamic_rnn(LSTMCell(name, n_in, n_hid), inputs, initial_state=h0, swap_memory=True)

def BiLSTM(
    name,
    inputs,
    n_in,
    n_hid,
    h0_1=None,
    h0_2=None
    ):
    batch_size = tf.shape(inputs)[0]
    if h0_1 is None:
        h0_1 = tflib.param(name+'.init.h0_1', np.zeros(2*n_hid, dtype='float32'))
        h0_1 = tf.reshape(tf.tile(h0_1, tf.pack([batch_size])), tf.pack([batch_size, 2*n_hid]))

    if h0_2 is None:
        h0_2 = tflib.param(name+'.init.h0_2', np.zeros(2*n_hid, dtype='float32'))
        h0_2 = tf.reshape(tf.tile(h0_2, tf.pack([batch_size])), tf.pack([batch_size, 2*n_hid]))


    cell1 = LSTMCell(name+'_fw', n_in, n_hid)
    cell2 = LSTMCell(name+'_bw', n_in, n_hid)

    seq_len = tf.tile(tf.expand_dims(tf.shape(inputs)[1],0),[batch_size])
    outputs = tf.nn.bidirectional_dynamic_rnn(cell1, cell2, inputs, sequence_length=seq_len, initial_state_fw=h0_1, initial_state_bw=h0_2, swap_memory=True)
    return tf.concat(2,[outputs[0][0],outputs[0][1]])

'''
Attention Mechanism as proposed in the Show, Attend and Tell paper (https://arxiv.org/pdf/1502.03044v3.pdf)
'''
class LSTMAttentionCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, name, n_in, n_hid, ctx, dim_ctx, batch_size=32, forget_bias=1.0):
        self._n_in = n_in
        self._n_hid = n_hid
        self._name = name
        self._forget_bias = forget_bias
        self._dim_ctx = dim_ctx
        self._batch_size = batch_size
        self._ctx = ctx

    @property
    def state_size(self):
        return self._n_hid

    @property
    def output_size(self):
        return self._n_hid

    def __call__(self, _input, state, scope=None):

        h_tm1, c_tm1 = tf.split(1,2,state)

        beta = tf.nn.sigmoid(tflib.ops.Linear(self._name+'._beta',h_tm1,self._n_hid,1))
        h_att = T.tanh(self._ctx + tf.expand_dims(tflib.ops.Linear(self._name+'.h_att',h_tm1,self._n_hid,self._dim_ctx[1]),1))
        e_t = tflib.ops.Linear('f_att',h_att,self._dim_ctx[1],1,bias=False)[:,:,0]
        alpha_t = tf.nn.softmax(e_t)

        z_t = beta*tf.reduce_sum(tf.expand_dims(alpha_t,2)*self._ctx,1)

        gates = tflib.ops.Linear(
                self._name+'.Gates',
                tf.concat(1, [_input, h_tm1, z_t]),
                self._n_in + self._n_hid + self._dim_ctx[1],
                4 * self._n_hid
            )

        i_t,f_t,o_t,g_t = tf.split(1, 4, gates)

        c_t = tf.nn.sigmoid(f_t+self._forget_bias)*c_tm1 + tf.nn.sigmoid(i_t)*tf.tanh(g_t)
        h_t = tf.nn.sigmoid(o_t)*tf.tanh(c_t)
        new_state = tf.concat(1,[h_t,c_t])

        new_out = tf.concat(1,[z_t,_input,h_t])

        return new_out,new_state

class OutputProjectionWrapper(tf.nn.rnn_cell.RNNCell):

    def __init__(self, cell, output_size):
        if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
            raise TypeError("The parameter cell is not RNNCell.")
        if output_size < 1:
            raise ValueError("Parameter output_size must be > 0: %d." % output_size)
        self._cell = cell
        self._output_size = output_size

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):
        """Run the cell and output projection on inputs, starting from state."""

        concat_out,state = self._cell(inputs,state)
        batch_size = tf.shape(concat_out)[0]
        c = self._cell
        z_t = tf.slice(concat_out,[0,0],[batch_size,c._dim_ctx[1]])
        _input = tf.slice(concat_out,[0,c._dim_ctx[1]],[batch_size,c._n_hid])
        h_t = tf.slice(concat_out,[0,c._dim_ctx[1]+c._n_hid],[batch_size,c._n_hid])

        # h_out = tflib.ops.Linear('h_out',h_t,c._n_hid,c._n_hid)
        # h_out += tflib.ops.Linear('ctx_out',z_t,c._dim_ctx[1],c._n_hid)
        # h_out += tflib.ops.Linear('prev2out',_input,c._n_hid,c._n_hid)
        # h_out = tf.tanh(h_out)

        h_out = tf.tanh(tflib.ops.Linear(
            'DeepOutput',
            tf.concat(1,[h_t,z_t,_input]),
            c._n_hid + c._dim_ctx[1] + c._n_hid,
            c._n_hid
        ))

        out_logits = tflib.ops.Linear('out_logits',h_out,c._n_hid,self._output_size)

        return out_logits, state

def LSTMAttention(
    name,
    inputs,
    context,
    n_in,
    n_hid,
    n_out,
    dim_ctx,
    h0,
    reset
    ):
    print h0
    def f1():
        return tf.tanh(tflib.ops.Linear(name+'.Init.ch',tf.reduce_mean(context,1),dim_ctx[-1],2*n_hid))
    def f2():
        return h0
    init = tf.cond(tf.equal(reset[0],1),f1, f2)
    ctx_proj = tflib.ops.Linear('Project.Features',context,dim_ctx[1],dim_ctx[1],bias=False)
    cell = LSTMAttentionCell(name, n_in, n_hid, ctx_proj, dim_ctx)
    out_cell = OutputProjectionWrapper(cell,n_out)
    out = tf.nn.dynamic_rnn(out_cell, inputs, initial_state=init, swap_memory=True)
    return out[0], out[1]

'''
Attentional Decoder as proposed in HarvardNLp paper (https://arxiv.org/pdf/1609.04938v1.pdf)
'''
class im2latexAttentionCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, name, n_in, n_hid, L, D, ctx, forget_bias=1.0):
        self._n_in = n_in
        self._n_hid = n_hid
        self._name = name
        self._forget_bias = forget_bias
        self._ctx = ctx
        self._L = L
        self._D = D

    @property
    def state_size(self):
        return self._n_hid

    @property
    def output_size(self):
        return self._n_hid

    def __call__(self, _input, state, scope=None):

        h_tm1, c_tm1, output_tm1 = tf.split(1,3,state)

        gates = tflib.ops.Linear(
                self._name+'.Gates',
                tf.concat(1, [_input, output_tm1]),
                self._n_in + self._n_hid,
                4 * self._n_hid,
                activation='sigmoid'
            )

        i_t,f_t,o_t,g_t = tf.split(1, 4, gates)

        ## removing forget_bias
        c_t = tf.nn.sigmoid(f_t)*c_tm1 + tf.nn.sigmoid(i_t)*tf.tanh(g_t)
        h_t = tf.nn.sigmoid(o_t)*tf.tanh(c_t)


        target_t = tf.expand_dims(tflib.ops.Linear(self._name+'.target_t',h_t,self._n_hid,self._n_hid,bias=False),2)
        # target_t = tf.expand_dims(h_t,2) # (B, HID, 1)
        a_t = tf.nn.softmax(tf.batch_matmul(self._ctx,target_t)[:,:,0]) # (B, H*W, D) * (B, D, 1)

        a_t = tf.expand_dims(a_t,1) # (B, 1, H*W)
        z_t = tf.batch_matmul(a_t,self._ctx)[:,0]
        # a_t = tf.expand_dims(a_t,2)
        # z_t = tf.reduce_sum(a_t*self._ctx,1)

        output_t = tf.tanh(tflib.ops.Linear(
            self._name+'.output_t',
            tf.concat(1,[h_t,z_t]),
            self._D+self._n_hid,
            self._n_hid,
            bias=False,
            activation='tanh'
            ))

        new_state = tf.concat(1,[h_t,c_t,output_t])

        return output_t,new_state

'''
Wrapper function for Bidirectional LSTM encoder on CNN features grid + Attentional Decoder as proposed in the HarvardNLP paper (https://arxiv.org/pdf/1609.04938v1.pdf)
'''
def im2latexAttention(
    name,
    inputs,
    ctx,
    input_dim,
    ENC_DIM,
    DEC_DIM,
    D,
    H,
    W
    ):

    V = tf.transpose(ctx,[0,2,3,1]) # (B, H, W, D)
    V_cap = []
    batch_size = tf.shape(ctx)[0]
    count=0

    h0_i_1 = tf.tile(tflib.param(
        name+'.Enc_.init.h0_1',
        np.zeros((1,H,2*ENC_DIM)).astype('float32')
    ),[batch_size,1,1])

    h0_i_2 = tf.tile(tflib.param(
        name+'.Enc_init.h0_2',
        np.zeros((1,H,2*ENC_DIM)).astype('float32')
    ),[batch_size,1,1])


    def fn(prev_out,i):
    # for i in xrange(H):
        return tflib.ops.BiLSTM(name+'.BiLSTMEncoder',V[:,i],D,ENC_DIM,h0_i_1[:,i],h0_i_2[:,i])

    V_cap = tf.scan(fn,tf.range(tf.shape(V)[1]), initializer=tf.placeholder(shape=(None,None,2*ENC_DIM),dtype=tf.float32))

    V_t = tf.reshape(tf.transpose(V_cap,[1,0,2,3]),[tf.shape(inputs)[0],-1,ENC_DIM*2]) # (B, L, ENC_DIM)

    h0_dec = tf.tile(tflib.param(
        name+'.Decoder.init.h0',
        np.zeros((1,3*DEC_DIM)).astype('float32')
    ),[batch_size,1])

    cell = tflib.ops.im2latexAttentionCell(name+'.AttentionCell',input_dim,DEC_DIM,H*W,2*ENC_DIM,V_t)
    seq_len = tf.tile(tf.expand_dims(tf.shape(inputs)[1],0),[batch_size])
    out = tf.nn.dynamic_rnn(cell, inputs, initial_state=h0_dec, sequence_length=seq_len, swap_memory=True)

    return out

def Embedding(
    name,
    n_symbols,
    output_dim,
    indices
    ):
    with tf.name_scope(name) as scope:
        emb = tflib.param(
            name,
            initializer('Normal',[n_symbols, output_dim],std=1.0/np.sqrt(n_symbols))
            )

        return tf.nn.embedding_lookup(emb,indices)
