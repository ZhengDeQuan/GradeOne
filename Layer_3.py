#!/usr/bin/python
# -*- coding:UTF-8 -*-
import os
import re
import pickle
import numpy as np
import keras.backend as K
from keras.layers import Input, Embedding, LSTM, Dense, Merge, merge, AveragePooling1D,GlobalAveragePooling1D
from keras.layers import Activation, Reshape, TimeDistributed, Lambda, Conv1D, Dropout,MaxPooling1D
from keras.layers import Permute, RepeatVector
from keras.models import Model

from keras.engine.topology import Layer, InputSpec
from keras import backend as T
#from Layer_Data import get_data, get_Med_data, get_Padded_data, get_triple, make_fake_y,make_fake_y2
from Layer_Data import *
from Layer_1 import TemporalMeanPooling
from Layer_1 import Step_One
from Layer_2 import *


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

#获得首次的train数据
Train_data = get_data(indir,train_file)
Med_Train_data = get_Med_data(Train_data)#get Manipulated data
Padded_Train_data = get_Padded_data(Med_Train_data, MAX_SEQUENCE_LENGTH)
Que, Ans1, Ans2 = get_triple(Padded_Train_data)
X = [Que,Ans1,Ans2]

#获得验证集的数据
Valid_data = get_data(indir,valid_file)
Med_Valid_data = get_Med_data_for_test(Valid_data)
Padded_Valid_data = get_Padded_data_for_test(Med_Valid_data, MAX_SEQUENCE_LENGTH)
Que_V, Ans1_V, Ans2_V = get_triple_for_test(Padded_Valid_data)
qids_V = get_qids(Valid_data)

#定义模型的部分
question = Input(shape=(MAX_SEQUENCE_LENGTH,))
answer_good = Input(shape=(MAX_SEQUENCE_LENGTH,))
answer_bad = Input(shape=(MAX_SEQUENCE_LENGTH,))

cos12= Step_Two([question , answer_good])
cos13 = Step_Two([question , answer_bad])

def myclip(cos12,cos13,Margin):
    return K.clip(x = Margin-cos12+cos13,min_value= 0.0 , max_value=None)

out = Lambda(myclip,arguments={'cos13':cos13,'Margin':Margin})(cos12)

#cost = Lambda(K.sum,arguments={'axis':0})(out)
#cost = Lambda(K.sum)(out)#如果是用sum的话,会不会在最后不足一个batch的组上梯度比较小，mean是不是更合理？

def myloss(y_true,y_pred):
    return K.sum(y_pred,axis=0,keepdims=False)
# 这样写不行，好像要加到losses.py文件中

# def mean_absolute_error(y_true, y_pred):
#     return K.mean(K.abs(y_pred - y_true), axis=-1)
# 这个是库中提供的,我每次传入的y_true，都是全0的向量就行了,只可惜他是mean，原来的代码中是sum

# def zq_mean_absolute_error(y_true, y_pred):
#     return K.sum(K.abs(y_pred - y_true) , axis=-1)
# 这个我写到了losses.py中
Step_Three = Model(inputs=[question,answer_good,answer_bad],outputs=out)
Step_Test = Model(inputs = [question,answer_good,answer_bad],outputs=cos12)#预测的时候用的模型
Step_Three.compile(optimizer='sgd',loss='mae')
Y = make_fake_y(len(Que))
last_MAP,last_MRR = 0.0,0.0
present_MAP,present_MRR = 0.0,0.0
weights_now = []
weights_before = None
for epoch in range(8):
    #因为在get_Med_data中，有随机数函数，所以每次产生的wrong_answer数据都有可能不同
    Med_Train_data = get_Med_data(Train_data)  # get Manipulated data
    Padded_Train_data = get_Padded_data(Med_Train_data, MAX_SEQUENCE_LENGTH)
    Que, Ans1, Ans2 = get_triple(Padded_Train_data)
    Step_Three.fit(x=[Que,Ans1,Ans2],y=Y)
    weights_now = Step_Three.get_weights()
    if weights_before == None :
        weights_before = None
    else:
        weights_before = Step_Test.get_weights()
    Step_Test.set_weights(weights_now)
    Y_test1 = Step_Test.predict(x=[Que_V,Ans1_V,Ans2_V])
    sdict = make_sdict(Y_test1,qids_V)
    present_MAP = cal_MAP(sdict)
    present_MRR = cal_MRR(sdict)
    print("in epoch :",epoch)
    print("valid_score :")
    print("present_MAP = ",present_MAP)
    print("present_MRR = ",present_MRR)
    if (last_MAP > present_MAP) or (last_MAP == present_MAP and last_MRR > present_MRR):
        Step_Test.set_weights(weights_before)
        print("Updated")
    else :
        #新的权重更好
        weights_before = weights_now
        last_MAP = present_MAP
        last_MRR = present_MRR


# Y_p = Step_Three.predict(x = [Que,Ans1,Ans2])
# print("Y_p = ",Y_p)
# 只是试验一下好不好用

# weights_now = Step_Three.get_weights()
# print("weights = ",weights_now)
# Step_Test.set_weights(weights_now)
# 不一定使用最后一次的权重了
Test_data = get_data(indir,test_file)
Med_Test_data = get_Med_data_for_test(Test_data)
Padded_Test_data = get_Padded_data_for_test(Med_Test_data, MAX_SEQUENCE_LENGTH)
Que_T, Ans1_T, Ans2_T = get_triple_for_test(Padded_Test_data)
qids_T = get_qids(Test_data)
Y_test1 = Step_Test.predict(x=[Que_T,Ans1_T,Ans2_T])
sdict = make_sdict(Y_test1,qids_T)
MAP = cal_MAP(sdict)
MRR = cal_MRR(sdict)
print("test_score :")
print("MAP = ",MAP)
print("MRR = ",MRR)

