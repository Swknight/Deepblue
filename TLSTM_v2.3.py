#coding=utf-8
import TLSTMCell_v2 as TLSTMCell
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.contrib import rnn
# import pandas as pd
import numpy as np
import random

embedding_length = 100


class TLSTM(object):
    _reg_scope = 'reg_col'

    def __init__(self,
                 batch_size=32,
                 lr=0.008,
                 mode='train',
                 input_n_layer=1,
                 input_max_len=40,
                 n_layers=1,
                 time_step_input=94,
                 time_step_twitter=6):
        self._input_n_layer = input_n_layer
        self._input_max_len = input_max_len
        self._batch_size = batch_size

        self._lr = lr
        self._time_step_input = time_step_input
        self._time_step_twitter = time_step_twitter
        self._n_layers = n_layers
        self._mode = mode

    def _build_graph(self):

        def _get_TLstm_cells(hidden_layer):
            return TLSTMCell.TLSTMCell(num_units=hidden_layer)

        def _get_inputLstm_cells(hidden_layer):
            return tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_layer)

        # input lstm
        with tf.variable_scope('net'):
            # input the twitter sentence
            self.inputs = tf.placeholder(dtype=tf.float32, shape=(
            self._batch_size, self._time_step_twitter + 1, self._time_step_input, embedding_length), name="inputs")
            # input the length of twitter sentence
            self.inputs_sequence = tf.placeholder(dtype=tf.float32,
                                                  shape=(self._batch_size, self._time_step_twitter + 1),
                                                  name="inputs_sequence")
            # input the number of a time sequence
            self.time_sequence = tf.placeholder(dtype=tf.float32, shape=(self._batch_size), name="time_sequence")
            # input the reply of each twitter
            self.targets = tf.placeholder(dtype=tf.float32, shape=(self._batch_size, self._time_step_twitter),
                                          name="targets")
            # input the delt time of each two twitters.
            self.delt_time = tf.placeholder(dtype=tf.float32, shape=(self._batch_size, self._time_step_twitter, 1),
                                            name="delt_time")
            # input the predict reply
            self.y = tf.placeholder(dtype=tf.float32, shape=(self._batch_size, 1))
            self.inputs_keep_prob = tf.placeholder(dtype=tf.float32)
            self.tlstm_keep_prob = tf.placeholder(dtype=tf.float32)

        with tf.variable_scope('Inputs_LSTM'):
            input_cells_list = []
            input_hidden_shape = 128
            for num in range(self._input_n_layer):
                input_cells_list.append(_get_inputLstm_cells(hidden_layer=input_hidden_shape))
            for ind_input_lstm, each_input_lstm in enumerate(input_cells_list):
                input_cells_list[ind_input_lstm] = tf.contrib.rnn.DropoutWrapper(each_input_lstm,
                                                                                 output_keep_prob=self.inputs_keep_prob)
            input_cells = tf.contrib.rnn.MultiRNNCell(input_cells_list)
            input_cells_state = input_cells.zero_state(self._batch_size, tf.float32)
            inputs_sentence = []

            for _ in range(self._time_step_twitter + 1):
                input_outputs, input_last_states = tf.nn.dynamic_rnn(input_cells, inputs=self.inputs[:, _, :, :],
                                                                     dtype=tf.float32,
                                                                     sequence_length=self.inputs_sequence[:, _],
                                                                     initial_state=input_cells_state,
                                                                     )
                inputs_sentence.append(input_last_states[-1].h)
            inputs_sentence = tf.convert_to_tensor(inputs_sentence)

        # TLstm_y
        with tf.variable_scope('TLSTM_Y'):
            cells_list = []
            # self.targets = tf.reshape(self.targets,[self._batch_size,self._time_step_twitter,1])
            # self.delt_time = tf.reshape(self.delt_time,[self._batch_size,self._time_step_twitter,1])
            hidden_shape = 128
            for num in range(self._n_layers):
                cells_list.append(_get_TLstm_cells(hidden_layer=hidden_shape))
            for ind_lstm, each_lstm in enumerate(cells_list):
                cells_list[ind_lstm] = tf.contrib.rnn.DropoutWrapper(each_lstm, output_keep_prob=self.tlstm_keep_prob)
            cells = tf.contrib.rnn.MultiRNNCell(cells_list)
            cells_state = cells.zero_state(self._batch_size, tf.float32)
            outputs0, last_states0 = tf.nn.dynamic_rnn(cells, tf.reshape(tf.concat(
                [tf.reshape(self.delt_time, [self._batch_size, self._time_step_twitter, 1]),
                 tf.reshape(self.targets, [self._batch_size, self._time_step_twitter, 1]), ], 2),
                                                                         [-1, self._time_step_twitter, 2]),
                                                       initial_state=cells_state, sequence_length=self.time_sequence,
                                                       # dtype = tf.float32,
                                                       )
            # output_final_y = last_states0[-1].h

        with tf.variable_scope('TLSTM_X'):
            cells_list = []
            hidden_shape = 128
            for num in range(self._n_layers):
                cells_list.append(_get_TLstm_cells(hidden_layer=hidden_shape))
            for ind_lstm, each_lstm in enumerate(cells_list):
                cells_list[ind_lstm] = tf.contrib.rnn.DropoutWrapper(each_lstm, output_keep_prob=self.tlstm_keep_prob)
            cells = tf.contrib.rnn.MultiRNNCell(cells_list)
            cells_state = cells.zero_state(self._batch_size, tf.float32)

            # y_state_head,y_state_last = tf.convert_to_tensor(outputs0)[:,:-1,:],tf.convert_to_tensor(outputs0)[:,-1:,:]
            begin_state = tf.Variable(tf.random_normal([hidden_shape]), name='begin_state')

            y_state = tf.concat(
                [tf.convert_to_tensor([[begin_state]] * self._batch_size), tf.convert_to_tensor(outputs0)], 1)

            input_delt_time = tf.concat([tf.zeros([self._batch_size, 1, 1], dtype=tf.float32),
                                         tf.reshape(self.delt_time, [self._batch_size, self._time_step_twitter, 1])], 1)

            tmp = tf.concat([input_delt_time, tf.transpose(inputs_sentence, [1, 0, 2]), tf.convert_to_tensor(y_state)],
                            2)

            outputs0, last_states0 = tf.nn.dynamic_rnn(cells, tf.reshape(tmp,
                                                                         [self._batch_size, self._time_step_twitter + 1,
                                                                          -1]),
                                                       initial_state=cells_state,
                                                       # dtype = tf.float32,
                                                       )
            output_final = last_states0[-1].h

        with tf.variable_scope('fcl'):
            weights = {
                # 'mat1': tf.Variable(tf.random_normal([timesteps*num_hidden, 200])),
                'fc1': tf.Variable(tf.random_normal([hidden_shape, 100]), name='w1'),
                'fc2': tf.Variable(tf.random_normal([100, 1]), name='w2'),
            }
            biases = {
                'fc1': tf.Variable(tf.random_normal([100]), name='b1'),
                'fc2': tf.Variable(tf.random_normal([1]), name='b2'),
            }
            # output_final = self._linear_connect(output_final,output_s=64,activation = tf.nn.relu) #output_s
            # output_logit = self._linear_connect(output_final, output_s=1, activation=None) #output_s s
            x_matmumat1 = tf.matmul(output_final, weights['fc1']) + biases['fc1']
            output_logit = tf.matmul(x_matmumat1, weights['fc2']) + biases['fc2']

        self.loss = tf.nn.log_poisson_loss(log_input=output_logit, targets=self.y)
        self.mse = tf.nn.l2_loss(math_ops.exp(output_logit)-self.y)
        tf.add_to_collection(self._reg_scope, self.loss)
        self.action_reg_loss = tf.add_n(tf.get_collection(self._reg_scope), name='action_reg_loss')
        self.action_passion_loss = tf.summary.scalar(name='action_passion_loss',
                                                     tensor=self.action_reg_loss)  # 泊松回归的loss值

        if self._mode == 'train':
            adam_optimeizer = tf.train.AdamOptimizer(learning_rate=self._lr)
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            raw_grad = tf.gradients(self.loss, train_vars)
            clip_grad, _ = tf.clip_by_global_norm(raw_grad,5)
            self.adam_train_op = adam_optimeizer.apply_gradients(zip(clip_grad, train_vars))

        self.action_predict_label = math_ops.exp(output_logit)  # 预测值

        self.init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(self.init)

            merged = tf.summary.merge_all()
            # tf.summary.FileWriter()将所有信息写入文件
            writer = tf.summary.FileWriter('./tensorflow1.1_logs', sess.graph)

    def train(self, epoch=10):
        print("-------------------------Load data-----------------------------------")
        # in_inputs_sequence = np.load('../data/lstm_train_inputs_sequence_0.25_nrl_except_outlier.npy')
        # in_targets = np.load('../data/lstm_train_targets_0.25_nrl_except_outlier.npy')
        # in_delt_time = np.load('../data/lstm_train_delt_time_0.25_nrl_except_outlier.npy')
        # in_y = np.load('../data/lstm_train_y_0.25_nrl_except_outlier.npy')
        # in_time_sequence = np.load('../data/lstm_train_time_sequence_0.25_nrl_except_outlier.npy').astype(np.float32)
        # in_inputs = np.load('../data/lstm_train_inputs_0.25_nrl_except_outlier.npy')
        # num_sample = 27054 # 0.25 samples

        # in_inputs_sequence= np.load('../data/lstm_train_inputs_sequence_nrl_except_outlier.npy')
        # in_targets= np.load('../data/lstm_train_targets_nrl_except_outlier.npy')
        # in_delt_time = np.load('../data/lstm_train_delt_time_nrl_except_outlier.npy')
        # in_y = np.load('../data/lstm_train_y_nrl_except_outlier.npy')
        # in_time_sequence = np.load('../data/lstm_train_time_sequence_nrl_except_outlier.npy').astype(np.float32)
        # in_inputs = np.load('../data/lstm_train_inputs_nrl_except_outlier.npy')
        # num_sample = int(108219) # total samples

        in_inputs_sequence= np.load('../data/lstm_train_inputs_sequence_maxmin_nrl_except_outlier_except0_0.5.npy')
        in_targets= np.load('../data/lstm_train_targets_maxmin_nrl_except_outlier_except0_0.5.npy')
        in_delt_time = np.load('../data/lstm_train_delt_time_maxmin_nrl_except_outlier_except0_0.5.npy')
        in_y = np.load('../data/lstm_train_y_maxmin_nrl_except_outlier_except0_0.5.npy')
        in_time_sequence = np.load('../data/lstm_train_time_sequence_maxmin_nrl_except_outlier_except0_0.5.npy').astype(np.float32)
        in_inputs = np.load('../data/lstm_train_inputs_maxmin_nrl_except_outlier_except0_0.5.npy')
        num_sample = int(32935) # total samples


        split = 0.8
        train_inputs_sequence = in_inputs_sequence[:int(num_sample * split)]
        train_targets = in_targets[:int(num_sample * split)]
        train_delt_time = in_delt_time[:int(num_sample * split)]
        train_y = in_y[:int(num_sample * split)]
        train_time_sequence = in_time_sequence[:int(num_sample * split)]
        train_inputs = in_inputs[:int(num_sample * split)]

        test_inputs_sequence = in_inputs_sequence[int(num_sample * split):]
        test_targets = in_targets[int(num_sample * split):]
        test_delt_time = in_delt_time[int(num_sample * split):]
        test_y = in_y[int(num_sample * split):]
        test_time_sequence = in_time_sequence[int(num_sample * split):]
        test_inputs = in_inputs[int(num_sample * split):]

        def data_loader(batch_size):
            idx = list(range(int(num_sample * split)))
            random.shuffle(idx)
            for i in range(0, int(num_sample * split), batch_size):
                j = np.array(idx[i:min(i + batch_size, int(num_sample * split))])
                if len(j) != batch_size:
                    break
                yield train_inputs_sequence[j], train_targets[j], train_delt_time[j], train_y[j], train_time_sequence[
                    j], train_inputs[j]

        with tf.Session() as sess:
            sess.run(self.init)
            epoch_num = 0
            for one_epoch in range(epoch):
                step = 1
                epoch_num += 1
                mean_loss = 0
                print("---------------------------EPOCH" + str(epoch_num) + "-----------------------")
                for oneepoch_data in data_loader(32):
                    __inputs_sequence, __targets, __delt_time, __y, __time_sequence, __inputs = oneepoch_data
                    adamTrain, lossValue,mse,prediction = sess.run([self.adam_train_op, self.loss,self.mse, self.action_predict_label], feed_dict={self.inputs: __inputs,
                                                                                                self.inputs_sequence: __inputs_sequence,
                                                                                                self.targets: __targets,
                                                                                                self.inputs_keep_prob: 0.8,
                                                                                                self.tlstm_keep_prob: 0.8,
                                                                                                self.delt_time: __delt_time,
                                                                                                self.y: __y,
                                                                                                self.time_sequence: __time_sequence})
                    mean_loss += int(mse)
                    print("---------------------------step" + str(step) + "-------loss" + str(mse))
                    # print ("---------------y" + str(__y))
                    # print("---------------prediction" + str(prediction))
                    # print("---------------target" + str(__targets))
                    step += 1
                    pass
                print("---------------------------Loss" + str(float(mean_loss/823)))
                # print(step,"")
            merged = tf.summary.merge_all()
            # tf.summary.FileWriter()将所有信息写入文件
            writer = tf.summary.FileWriter('./tensorflow1.1_logs', sess.graph)

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
    # def __init__(self,
    #          batch_size = 32,
    #          lr = 0.01,
    #          mode = 'train',
    #          input_n_layer = 1,
    #          input_max_len = 40,
    #          n_layers = 1,
    #          time_step = 7):
    model = TLSTM()
    print("-------------------------Build Graph---------------------------------")
    model._build_graph()
    print("-------------------------Start Training------------------------------")
    model.train(epoch=30)
