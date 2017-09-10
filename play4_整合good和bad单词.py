#!/usr/bin/python
# -*- coding:UTF-8 -*-
import os
import re
import pickle

indir = "NewCorpus6"
innerdir1 = "good"
innerdir2 = "bad"
outdir = "NewCorpus7"

offset = 26346+1
file1 = os.path.join(indir,innerdir1,"good_word2vec.pkl")
file2 = os.path.join(indir,innerdir2,"bad_word2vec.pkl")
f1 = open(file1,'rb')
f2 = open(file2,'rb')
good_word2vec = pickle.load(f1)
bad_word2vec = pickle.load(f2)
f1.close()
f2.close()





#
# id2word = {}
# for i_v in good_id2word.items():
#     print(i_v)
#     id2word[i_v[0]] = i_v[1]
#
# for i_v in bad_id2word.items():
#     id2word[i_v[0]+offset] = i_v[1]


#
# id2vec = {}
# for i_v in good_id2vec.items():
#     id2vec[i_v[0]] = i_v[1]
# for i_v in bad_id2vec.items():
#     id2vec[i_v[0]+offset] = i_v[1]
#
# import numpy as np
# id2vec_array = np.zeros((26346+1+305,300),'float32')
# for id in range(1,len(id2vec)):
#     id2vec_array[id] = id2vec[id]
#
# file_extra = os.path.join(indir,innerdir1,"good_id2vec_array.pkl")
# file_extra = open(file_extra,'rb')
# data = pickle.load(file_extra)
# id2vec_array[0] = data[0]
#


# word2id = {}
# for w_i in good_word2id.items():
#     word2id[w_i[0]] = w_i[1]
#
# for w_i in bad_word2id.items():
#     word2id[w_i[0]] = w_i[1]+offset

word2vec = {}
for w_v in good_word2vec.items():
    word2vec[w_v[0]] = w_v[1]
for w_v in bad_word2vec.items():
    word2vec[w_v[0]] = w_v[1]

file3 = os.path.join(outdir,"word2vec.pkl")
with open(file3,"wb") as f3:
    pickle.dump(word2vec,f3)


