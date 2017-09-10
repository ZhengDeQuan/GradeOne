#!/usr/bin/python
# -*- coding:UTF-8 -*-
#将id2vec_array中的0指引的向量，由model.seeded_vector('0')变成全零向量
import os
import pickle
import numpy as np

indir = 'NewCorpus7'
file = os.path.join(indir,'id2vec_array.pkl')
with open(file,'rb') as f:
    id2vec_array_0_zeros = pickle.load(f)

print("len = ",len(id2vec_array_0_zeros))
print(" 0 = ",id2vec_array_0_zeros[0])
id2vec_array_0_zeros[0] = np.zeros((300,))
print(" 0 = ",id2vec_array_0_zeros[0])

file2 = os.path.join(indir,'id2vec_array_0_zeros.pkl')
with open(file2,'wb') as f:
    pickle.dump(id2vec_array_0_zeros,f)
print("写完成")

