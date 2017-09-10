#!/usr/bin/python
# -*- coding:UTF-8 -*-

import os
import re
import pickle
import keras.backend as K
import numpy as np
from keras.layers import Input, Embedding, LSTM, Dense, Merge, merge, AveragePooling1D,GlobalAveragePooling1D
from keras.layers import Activation, Reshape, TimeDistributed, Lambda, Conv1D, Dropout,MaxPooling1D
from keras.layers import Permute, RepeatVector
from keras.models import Model

from keras.engine.topology import Layer, InputSpec
from keras import backend as T
from Layer_Data import get_data, get_Med_data, get_Padded_data, get_triple, make_fake_y,make_fake_y2,make_fake_y3
from Layer_1 import TemporalMeanPooling
from Layer_1 import Step_One
indir = "NewCorpus7"
# train_file = os.path.join(indir,'WikiQASent-train-filtered.txt')
# valid_file = os.path.join(indir,'WikiQASent-dev-filtered.txt')
# test_file = os.path.join(indir,'WikiQASent-test-filtered.txt')
train_file = 'WikiQASent-train-filtered.txt'
valid_file = 'WikiQASent-dev-filtered.txt'
test_file = 'WikiQASent-test-filtered.txt'


EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 118
hidden_lstm = 300
hidden_attention = 300
Margin = 0.05

#定义第二层
sentence_que = Input(shape=(MAX_SEQUENCE_LENGTH,))
sentence_ans = Input(shape=(MAX_SEQUENCE_LENGTH,))
encoded_que = Step_One(sentence_que)
encoded_ans = Step_One(sentence_ans)
Oq = TemporalMeanPooling()(encoded_que)
# Oq = AveragePooling1D(pool_size=MAX_SEQUENCE_LENGTH)(encoded_que)
# Oq = Reshape((hidden_lstm*2 , ))(Oq)
'''
Attention Machenism:
maq(t) = tanh(Wam ha(t) + Wqm Oq)
saq(t) 正比于 exp(wms^T maq(t))
~ha(t) = ha(t) saq(t)
'''
Wqm_Oq = Dense(units = hidden_attention)(Oq)
Wqm_Oq_padded = RepeatVector(MAX_SEQUENCE_LENGTH)(Wqm_Oq)#将Wqm_Oq由(nb_samples,hidden_attention)-->(nb_samples,timesteps,hidden_attention),为了之后能进行Wqm_Oq + Wam_ha的操作
Wam = TimeDistributed(Dense(units=hidden_attention),input_shape = (MAX_SEQUENCE_LENGTH , hidden_lstm * 2))#MAX_SEQUENCE_LENGTH :TimeStep
Wam_ha = Wam(encoded_ans)
# def broadcast_add(Wam_ha,Wqm_Oq_padded,hidden_attention):
#     print("Wqm_Oq_padded = ",Wqm_Oq_padded)
#     print("Wam_ha = ",Wam_ha)
#     return Wam_ha + Wqm_Oq#应该是这个+操作不允许#上次Model3调用model2一直有问题，根源就在这里
#
# Wam_ha_Wqm_Oq = Lambda(broadcast_add,output_shape=(MAX_SEQUENCE_LENGTH,hidden_attention),{'Wqm_Oq_padded':Wqm_Oq_padded,'hidden_attention':hidden_attention})(Wam_ha)
Wam_ha_Wqm_Oq = merge([Wam_ha,Wqm_Oq_padded],mode='sum')
maq = Activation('tanh')(Wam_ha_Wqm_Oq)
wms = TimeDistributed(Dense(units=1),input_shape = (MAX_SEQUENCE_LENGTH,hidden_attention))
wms_maq = wms(maq)#shape = (1040.118,1)==(batch_size,max_sequence_len,1)
print("wms_maq = ",wms_maq)
#wms是一个(1,hidden_attention)shape的向量, maq是一个(max_sequence_length,hidden_sttention)的矩阵
#这个操作，希望wms与矩阵中的每一行的对应元素相乘
wms_maq = Reshape((MAX_SEQUENCE_LENGTH,))(wms_maq)
print("wms_maq = ",wms_maq)
saq = Activation('softmax')(wms_maq)
#softmax(x, axis=-1),softmax参数默认是这样的
print("saq = ",saq)
saq = RepeatVector(hidden_lstm * 2)(saq)#(?,600,118)
saq = Permute((2,1))(saq)#(?,118,600)
hat_ha = merge([encoded_ans,saq] , mode = 'mul',dot_axes = -1)
#hat_ha 跟ha同型，只是相当于ha中的每个timestep的向量乘了一个权重系数

#之后biAttention有两种方案

# encoded_que ,hat_ha,分别是问答的，新的向量
CNN_Tensor_Que = []
CNN_Tensor_Ans = []
num_filters = 500#看论文的源码定的，太长了吧1500维的
for m in ([2,3,5]):
    this_tensor = Conv1D(filters=num_filters,kernel_size = m , strides=1, padding='same',activation='tanh', use_bias=True)(encoded_que)
    this_tensor = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH,strides=None,padding='valid')(this_tensor)
    CNN_Tensor_Que.append(this_tensor)

for m in ([2,3,5]):
    this_tensor = Conv1D(filters=num_filters,kernel_size=m,strides=1,padding='same',activation='tanh',use_bias=True)(hat_ha)
    this_tensor = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH, strides=None, padding='valid')(this_tensor)
    CNN_Tensor_Ans.append(this_tensor)

out_CNN_que = merge(CNN_Tensor_Que,mode='concat',concat_axis=-1)
out_CNN_que = Reshape((1500,))(out_CNN_que)
print("out_CNN = ",out_CNN_que)
out_CNN_ans = merge(CNN_Tensor_Ans,mode='concat',concat_axis=-1)
out_CNN_ans = Reshape((1500,))(out_CNN_ans)
que_drop = Dropout(0.5)(out_CNN_que)
ans_drop = Dropout(0.5)(out_CNN_ans)
out_cos = merge([que_drop,ans_drop],mode='cos',dot_axes=-1)
out_cos = Reshape((1,))(out_cos)
print("out_cos = ",out_cos)


Step_Two = Model(inputs = [sentence_que,sentence_ans],outputs =out_cos)
#第二层定义结束


#加载数据
# Train_data = get_data(indir,train_file)
# Valid_data = get_data(indir,valid_file)
# Med_Train_data = get_Med_data(Train_data)#get Manipulated data
# Padded_Train_data = get_Padded_data(Med_Train_data, MAX_SEQUENCE_LENGTH)
# Que, Ans1, Ans2 = get_triple(Padded_Train_data)
# print("Que.shape=",Que.shape)
# print("Ans1.shape = ",Ans1.shape)
# Y = make_fake_y(len(Que))
# print("Y.shape = ",Y.shape)
# print("In Layer_2")
# Step_Two.compile(optimizer='sgd',loss='binary_crossentropy')
# Step_Two.fit(x = [Que,Ans1] ,y = Y ,batch_size=30)
# YY = Step_Two.predict(x = [Que,Ans1])
# print("YY = ",YY)
# print("YY.shape = ",YY.shape)
#
# weights = Step_Two.get_weights()
# print("weights = ",weights)




