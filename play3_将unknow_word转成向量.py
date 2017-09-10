#!/usr/bin/python
# -*- coding:UTF-8 -*-

import os
import re
import pickle
import gensim
import numpy as np

indir = "NewCorpus6"
outdir = "NewCorpus6"
innerdir = "bad"
filename = "unknow_word.txt"
infile = os.path.join(indir,filename)
outfile = os.path.join(outdir,"bad_word2vec.pkl")
bad_word2vec = {}
bad_word2id = {}
bad_id2vec = {}
bad_id2word = {}
model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary = True);
print("load_succeed\n")
if __name__ == "__main__":
    ids = 0
    with open(infile,'r',encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.split()
            #line[0]中是word line[1]中是word的出现次数
            bad_word2id[line[0]] = ids
            bad_id2word[ids]=line[0]
            bad_id2vec[ids] = bad_word2vec[line[0]] = model.seeded_vector(line[0])
            ids += 1
    print("ids = ",ids)
    print("len(bad_word2id) = ",len(bad_word2id))

    pklnames = ['bad_word2id.pkl','bad_id2word.pkl','bad_word2vec.pkl','bad_id2vec.pkl']

    with open(os.path.join(outdir,innerdir,'bad_word2id.pkl'),'wb') as f:
        pickle.dump(bad_word2id,f)

    with open(os.path.join(outdir,innerdir,'bad_id2word.pkl'),'wb') as f:
        pickle.dump(bad_id2word,f)

    with open(os.path.join(outdir,innerdir,'bad_word2vec.pkl'),'wb') as f:
        pickle.dump(bad_word2vec,f)

    with open(os.path.join(outdir,innerdir,'bad_id2vec.pkl'),'wb') as f:
        pickle.dump(bad_id2vec,f)

    bad_id2vec_array = np.zeros((len(bad_id2vec),300),'float32')
    for id in range(ids):
        bad_id2vec_array[id] = bad_id2vec[id]

    with open(os.path.join(outdir,innerdir,'bad_id2vec_array.pkl'),'wb') as f:
        pickle.dump(bad_id2vec_array,f)


