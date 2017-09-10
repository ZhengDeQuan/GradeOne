#!/usr/bin/python
# -*- coding:UTF-8 -*-

import gensim
import pickle
#移除了cPickle模块，可以使用pickle模块代替
import numpy
import os

infiles = ['WikiQASent-test-filtered.txt','WikiQASent-dev-filtered.txt','WikiQASent-train-filtered.txt']
indir = "NewCorpus6"
outdir = "NewCorpus6"

model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary = True);
print("load_succeed\n")
dic_word2num = {}
dic_word_not_int_model = {}
dic_word2vec = {}
for filename in infiles:
    filename = os.path.join(indir,filename)
    with open(filename,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split('\t')
            line = line[:2]
            que = line[0].split()
            ans = line[1].split()
            has_word_ont_in_dir = False
            que += ans;
            for word in que :
                if word not in dic_word2num:
                    dic_word2num[word] = 1
                else:
                    dic_word2num[word] += 1

                if word in model:
                    dic_word2vec[word] = model[word]
                else:
                    has_word_ont_in_dir = True
                    #print("word = ",word)
                    if word not in dic_word_not_int_model:
                        dic_word_not_int_model[word] = 1
                    else:
                        dic_word_not_int_model[word] += 1

            # if has_word_ont_in_dir == True:
            #     print(que)


i = 1
for word in dic_word2num:
    if word in dic_word2vec:
        print(word)
        i += 1
        if i == 10:
            break
print(len(dic_word_not_int_model))
print(len(dic_word2num))
for word in dic_word_not_int_model:
    print(word)


with open(os.path.join(outdir,"word_not_in_model.txt"),"w",encoding='utf-8') as fout:
    for word in dic_word_not_int_model:
        fout.write(word+'\n')


print(len(dic_word_not_int_model))
print(len(dic_word2num))
import pickle
innerdir = "good"
dic_word2num = sorted(dic_word2num.items(),reverse = True,key = lambda item:item[1])
for i in range(10):
    print(dic_word2num[i])

good_word2id = {}
good_word2vec = {}
good_id2word = {}
good_id2vec = {}
ids = 1
for word in dic_word2num:
    word = word[0]
    if word in dic_word2vec:
        good_word2id[ word ] = ids
        good_id2word[ ids ] = word
        good_word2vec[word] = good_id2vec[ids] = dic_word2vec[word]
        ids += 1

print("ids = ",ids)
print(type(dic_word2vec))
print("len(word2vec) = ",len(dic_word2vec))

import numpy as np

good_id2vec_array = np.zeros((len(good_id2word)+1,300),'float32')
for id in range(1,len(good_id2word)+1):
    good_id2vec_array[id] = good_id2vec[id]

good_id2vec_array[0] = model.seeded_vector('0')

import pickle

with open(os.path.join(outdir,innerdir,'good_word2id.pkl'),'wb') as f:
    pickle.dump(good_word2id,f)

with open(os.path.join(outdir,innerdir,'good_word2vec.pkl'),'wb') as f:
    pickle.dump(good_word2vec,f)

with open(os.path.join(outdir,innerdir,'good_id2vec.pkl'),'wb') as f:
    pickle.dump(good_id2vec,f)

with open(os.path.join(outdir,innerdir,'good_id2word.pkl'),'wb') as f:
    pickle.dump(good_id2word,f)

with open(os.path.join(outdir,innerdir,'good_id2vec_array.pkl'),'wb') as f:
    pickle.dump(good_id2vec_array,f)

