import TLSTMCell as TLSTMCell
import tensorflow as tf
from tensorflow.python.ops import math_ops

import pandas as pd
import numpy as np

embedding_length = 128


class TLSTM(object):
    def __init__(self,
                 batch_size = 32,
                 lr = 0.01,
                 mode = 'train',
                 input_n_layer = 3,
                 input_max_len = 40,
                 n_layers = 3,
                 time_step = 7):
        self._input_n_layer = input_n_layer
        self._input_max_len = input_max_len
        self._batch_size = batch_size
        self._lr = lr
        self._time_step = time_step
        self._n_layers = n_layers
        self._mode = mode

        with tf.variable_scope('net'):
            self.inputs = tf.placeholder(dtype=tf.float32,shape=(batch_size,time_step,embedding_length),name = "inputs")
            self.inputs_sequence = tf.placeholder(dtype=tf.float32,shape=(batch_size,1),name = "inputs_sequence")
            self.targets = tf.placeholder(dtype=tf.float32,shape=(batch_size,1),name="targets")
            self.inputs_keep_prob = tf.placeholder(dtype = tf.float32)
            self.tlstm_keep_prob = tf.placeholder(dtype=tf.float32)
            self._build_graph()

        self.action_passion_loss = tf.summary.scalar(name='action_passion_loss',tensor=self.action_reg_loss) # 泊松回归的loss值




    def _build_graph(self):

        def _get_TLstm_cells(hidden_layer):
            return TLSTMCell.TLSTMCell(num_units=hidden_layer)

        def _get_inputLstm_cells(hidden_layer):
            return tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_layer)


        #input lstm
        input_cells_list = []
        input_hidden_shape = 128
        for num in range(self._input_n_layer):
            input_cells_list.append(_get_inputLstm_cells(hidden_layer=input_hidden_shape))
        for ind_input_lstm,each_input_lstm in enumerate(input_cells_list):
            input_cells_list[ind_input_lstm] = tf.contrib.rnn.DropoutWrapper(each_input_lstm,output_keep_prob = self.inputs_keep_prob)
        input_cells = tf.contrib.rnn.MultiRNNCell(input_cells_list)
        input_cells_state = input_cells.zero_state(self._batch_size,tf.float32)
        input_outputs,input_last_states = tf.nn.dynamic_rnn(input_cells,inputs=self.inputs,sequence_length=self.inputs_sequence,initial_state=input_cells_state)

        #TLstm
        cells_list = []
        hidden_shape = 128
        for num in range(self._n_layers):
            cells_list.append(_get_TLstm_cells(hidden_layer=hidden_shape))
        for ind_lstm,each_lstm in enumerate(cells_list):
            cells_list[ind_lstm] = tf.contrib.rnn.DropoutWrapper(each_lstm,output_keep_prob = self.tlstm_keep_prob)
        cells = tf.contrib.rnn.MultiRNNCell(cells_list)
        cells_state = cells.zero_state(self._batch_size,tf.float32)
        outputs,last_states = tf.nn.dynamic_rnn(cells,input_last_states[-1].h,initial_state=cells_state)
        output_final = last_states[-1].h

        with tf.variable_scope('fcl'):
            output_final = self._linear_connect(output_final,output_s=64,activation = tf.nn.relu) #output_s 没定
            output_logit = self._linear_connect(output_final, output_s=1, activation=None) #output_s 没定
        self.loss = tf.nn.log_poisson_loss(log_input=output_logit,targets=self.targets)

        if self._mode == 'train':
            adam_optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            raw_grad = tf.gradients(self.loss,train_vars)
            clip_grad, _= tf.clip_by_global_norm(raw_grad,10)
            self.adam_train_op = adam_optimizer.apply_gradients(zip(clip_grad,train_vars))

        self.action_predict_label = math_ops.exp(output_logit) # 预测值


    def _linear_connect(self, input_v, output_s, activation, is_reg=True):
        self._linear_layer_counter += 1
        weight_name = 'fc_w_%d' % self._linear_layer_counter
        bias_name = 'fc_b_%d' % self._linear_layer_counter
        input_shape = int(input_v.shape[1])
        weight = tf.get_variable(weight_name, shape=(input_shape, output_s), dtype=tf.float32)
        bias = tf.get_variable(bias_name, shape=(output_s), dtype=tf.float32, initializer=tf.zeros_initializer())

        # regularization collection
        if is_reg:
            tf.add_to_collection(self._reg_scope, tf.contrib.layers.l2_regularizer(self._reg_ratio)(weight))

        logits = tf.matmul(input_v, weight) + bias
        if activation is None:
            return logits
        else:
            return activation(logits)

if __name__ == '__main__':
    model = TLSTM()