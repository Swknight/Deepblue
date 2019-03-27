import numpy as np
import csv
import time 
import pickle 
from keras.preprocessing import sequence


def data_loader(X,Y,batch_size):
    idx = list(range(len(data)))
    random.shuffle(idx)
    for i in range(0,len(data),batch_size):
        j = np.array(idx[i:min(i+batch_size,len(data))])
        yield np.take(X,j),np.take(Y,j)


week = {"Mon":1,"Tue":2,"Wed":3,"Thu":4,"Fri":5,"Sat":6,"Sun":7}
month = {"Dec":12,'Nov':11,'Jan':1}

def data_pre():
    time_step_input = 94
    time_step_twitter = 7
    with open('../data/preprocessing.csv','r') as f:
        reader = csv.reader(f)
        twitter = [raw for raw in reader]
        # change datetime mat
        for i in range(len(twitter)):
            tmp = str(twitter[i][3][-4:]) + '-' + str(month[twitter[i][3][4:7]]) + '-' +str(twitter[i][3][8:10]).replace(' ','') + ' ' + str(twitter[i][3][11:19])
            tmp = time.strptime(tmp, "%Y-%m-%d %H:%M:%S")
            twitter[i][3] = time.mktime(tmp)
        d = {}
        for raw in twitter:
            usr_id = raw[2]
            if usr_id in d.keys():
                d[usr_id].append(raw)
            else:
                d[usr_id] = [raw]



        # input the length of twitter sentence 
        # self.inputs_sequence = tf.placeholder(dtype=tf.float32,shape=(self._batch_size,self._time_step_twitter+1),name = "inputs_sequence")
        inputs_sequence = []

        # input the number of a time sequence 
        # self.time_sequence = tf.placeholder(dtype=tf.float32,shape=(self._batch_size),name = "time_sequence")
        time_sequence = []

        # input the reply of each twitter 
        # self.targets = tf.placeholder(dtype=tf.float32,shape=(self._batch_size,self._time_step_twitter,1),name="targets")
        targets = [] 

        # input the delt time of each two twitters.
        # self.delt_time = tf.placeholder(dtype=tf.float32,shape=(self._batch_size,self._time_step_twitter,1),name="delt_time")
        delt_time = [] 

        # input the predict reply
        # self.y = tf.placeholder(dtype = tf.float32,shape = (self._batch_size,1))
        y = []

        # self.inputs = tf.placeholder(dtype=tf.float32,shape=(self._batch_size,self._time_step_twitter+1,self._time_step_input,embedding_length),name = "inputs")
        # inputs = []
        inputs = np.zeros([125964,time_step_twitter,time_step_input,100])
        cnt = 0

        f = open('../data/model', 'rb')
        model = pickle.load(f)

        f = open('../data/word_frequence', 'rb')
        word_freqs = pickle.load(f)
        zero_sum = 126000
        total_zero_sum = 0
        for key in d.keys():
            if len(d[key]) == 1:
                continue

            normalization = np.array([i[-1] for i in d[key]],dtype=np.float32)

        #均值归一化
            # zero = normalization == 0
            # if zero.sum() == normalization.shape[0]:
            #     total_zero_sum += 1

            # if zero_sum >0 and zero.sum() == normalization.shape[0]:
            #     zero_sum -= 1
            #     continue
            # if normalization.std() != 0:
            #     normalization = (normalization - normalization.mean())/normalization.std()
            # else:
            #     if normalization[0] > 0:
            #         print("std = 0,then ",normalization[0])
            #         normalization = np.ones([normalization.shape[0]])
            # # if sum(normalization) > normalization.shape[0]:
            # #     continue
            # outlier = normalization > 3
            # if outlier.sum() > 0:
            #     continue

        #最大最小值归一化
            if normalization.max() == 0:
                continue
            else:
                normalization = (normalization - normalization.min()) / (normalization.max() - normalization.min())



            for i in range(len(d[key])):
                d[key][i][-1] = normalization[i]

            for i in range(0,len(d[key]),time_step_twitter):
                sample = d[key][i:min(i+time_step_twitter,len(d[key]))]
                pad_number = time_step_twitter-len(sample)
                
                # no need padding
                y.append([float(sample[-1][-1])])
                time_sequence.append(len(sample))

                # need padding
                targets.append([float(item[-1]) for item in sample[:-1]] + [0]*pad_number)

                tmp = np.zeros([time_step_twitter,time_step_input,100])
                for i,item in enumerate(sample):
                    for j,word in enumerate(item[1].split(' ')):
                        if word_freqs[word] < 10:
                            continue
                        # tmp[i][j][:] = model[word]
                        inputs[cnt][i][j][:] = model[word]

                cnt += 1
                inputs_sequence.append([len(item[1].split(' ')) for item in sample] + [0]*pad_number)

                delt_time.append([[(int(sample[i][3]) - int(sample[i-1][3]))/3600000]  for i in range(1,len(sample))] + [[0]]*pad_number)
        y = np.array(y)
        time_sequence = np.array(time_sequence)
        targets = np.array(targets)
        inputs_sequence = np.array(inputs_sequence)
        delt_time = np.array(delt_time)
        print(y[:int(y.shape[0])].shape)
        print(time_sequence[:int(y.shape[0])].shape)
        print(targets[:int(y.shape[0])].shape)
        print(inputs[:int(y.shape[0])].shape)
        print(inputs_sequence[:int(y.shape[0])].shape)
        print(delt_time[:int(y.shape[0])].shape)
        print("total_zero" + str(total_zero_sum))
        np.save('../data/lstm_train_y_maxmin_nrl_except_outlier_except0.npy',y[:int(y.shape[0])])
        np.save('../data/lstm_train_time_sequence_maxmin_nrl_except_outlier_except0.npy',time_sequence[:int(y.shape[0])])
        np.save('../data/lstm_train_targets_maxmin_nrl_except_outlier_except0.npy',targets[:int(y.shape[0])])
        np.save('../data/lstm_train_inputs_maxmin_nrl_except_outlier_except0.npy',inputs[:int(y.shape[0])])
        np.save('../data/lstm_train_inputs_sequence_maxmin_nrl_except_outlier_except0.npy',inputs_sequence[:int(y.shape[0])])
        np.save('../data/lstm_train_delt_time_maxmin_nrl_except_outlier_except0.npy',delt_time[:int(y.shape[0])])
        print("data saved!")
        return y,time_sequence,targets,inputs,inputs_sequence,delt_time


         

if __name__ == "__main__":
    # y,time_sequence,targets,inputs,inputs_sequence,delt_time = data_loader.data_pre()
    y,time_sequence,targets,inputs,inputs_sequence,delt_time = data_pre()

    pass