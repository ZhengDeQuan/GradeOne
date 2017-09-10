#!/usr/bin/python
# -*- coding:UTF-8 -*-

import os
import re
import pickle
import keras.backend as K
import numpy as np
from keras.layers import Input, Embedding, LSTM, Dense, Merge, merge, AveragePooling1D,GlobalAveragePooling1D
from keras.layers import Activation, Reshape, TimeDistributed, Lambda, Conv1D, Dropout,MaxPooling1D
from keras.models import Model
from Layer_Data import get_data, get_Med_data, get_Padded_data, get_triple, make_fake_y,make_fake_y2
from keras.engine.topology import Layer, InputSpec
from keras import backend as T
K.set_learning_phase(1)

class TemporalMeanPooling(Layer):
    """
    This is a custom Keras layer. This pooling layer accepts the temporal
    sequence output by a recurrent layer and performs temporal pooling,
    looking at only the non-masked portion of the sequence. The pooling
    layer converts the entire variable-length hidden vector sequence
    into a single hidden vector, and then feeds its output to the Dense
    layer.

    input shape: (nb_samples, nb_timesteps, nb_features)
    output shape: (nb_samples, nb_features)
    """
    def __init__(self, **kwargs):
        super(TemporalMeanPooling, self).__init__(**kwargs)
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def call(self, x, mask=None): #mask: (nb_samples, nb_timesteps)
        if mask is None:
            mask = T.mean(T.ones_like(x), axis=-1)
        ssum = T.sum(x,axis=-2) #(nb_samples, np_features)
        mask = T.cast(mask,T.floatx())
        rcnt = T.sum(mask,axis=-1,keepdims=True) #(nb_samples)
        return ssum/rcnt
        #return rcnt

    def compute_mask(self, input, mask):
        return None


indir = "NewCorpus7"
filename = os.path.join(indir,"id2vec_array_0_zeros.pkl")
with open(filename,'rb') as f:
    embedding_matrix = pickle.load(f)
print("len(embedding_matrix)=",len(embedding_matrix))
word_index = len(embedding_matrix)

# train_file = os.path.join(indir,'WikiQASent-train-filtered.txt')
# valid_file = os.path.join(indir,'WikiQASent-dev-filtered.txt')
# test_file = os.path.join(indir,'WikiQASent-test-filtered.txt')
train_file = 'WikiQASent-train-filtered.txt'
valid_file = 'WikiQASent-dev-filtered.txt'
test_file = 'WikiQASent-test-filtered.txt'


EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 118
#MAX_SEQUENCE_LENGTH = 10
hidden_lstm = 300
hidden_attention = 300
Margin = 0.05

#定义Step_One Layer
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
print("sentence_input = ",sequence_input)
embedding_layer = Embedding(input_dim=word_index,
                            output_dim=EMBEDDING_DIM,
                            weights = [embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)#,mask_zero=True这个参数被抹掉了

embedded_sequence = embedding_layer(sequence_input)
print("embedded_sequence = ",embedded_sequence)
# forwards = LSTM(output_dim = hidden_lstm, return_sequences=True)(embedded_sequence)
#
# backwards = LSTM(output_dim = hidden_lstm,return_sequences=True,
#                  go_backwards=True)(embedded_sequence)

#merged = merge([forwards,backwards], mode = 'concat' , concat_axis= -1)#按照最后一个维度拼接
# Merge is for model
# merge is for tensor

forwards_LSTM = LSTM(output_dim = hidden_lstm,return_sequences= True)(embedded_sequence)
backwards_LSTM = LSTM(output_dim = hidden_lstm,return_sequences=True,go_backwards=True)(embedded_sequence)
merged = Merge(mode='concat',concat_axis=-1)([forwards_LSTM , backwards_LSTM])

print("merged = ",merged)

Step_One = Model(inputs = sequence_input, outputs= merged)#这个就是第一层的BiLSTM层
#第一层的定义结束


Train_data = get_data(indir,train_file)
Valid_data = get_data(indir,valid_file)
Med_Train_data = get_Med_data(Train_data)#get Manipulated data
Padded_Train_data = get_Padded_data(Med_Train_data,MAX_SEQUENCE_LENGTH)
Que , Ans1 , Ans2 = get_triple(Padded_Train_data)



#Step_One.compile(optimizer='sgd',loss='binary_crossentropy')
# Y = make_fake_y(len(Que))
# Y = make_fake_y2(len(Que))
# print("In Layer_1.py")
#Step_One.fit(epochs=1,x=Que,y=Y)