#!/usr/bin/python
# -*- coding:UTF-8 -*-

import os
import re
import pickle


indir = "NewCorpus7"
# file = os.path.join(indir,"id2vec_array.pkl")
# with open(file,'rb') as f:
#     id2vec_array = pickle.load(f)
#
# indexes = len(id2vec_array)
# print("i=",indexes)
file = os.path.join("NewCorpus7","word2id.pkl")
with open(file,"rb") as f:
    word2id = pickle.load(f)

indir = "NewCorpus6"
outdir = "NewCorpus7"
filenames = ["WikiQASent-dev-filtered.txt","WikiQASent-test-filtered.txt","WikiQASent-train-filtered.txt"]
for filename in filenames:
    infile = os.path.join(indir,filename)
    outfile = os.path.join(outdir,filename)
    with open(infile,"r",encoding='utf-8') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        line = line.split('\t')
        new_line = []
        for i in range(2):
            sentence = line[i].split()
            sentence_id = []
            for word in sentence:
                sentence_id.append(str(word2id[word]))
            sentence_id = " ".join(sentence_id)
            new_line.append(sentence_id)
        new_line.append(line[2])
        new_line.append(line[3])
        new_line = "\t".join(new_line)
        new_lines.append(new_line)
    with open(outfile,"w",encoding='utf-8') as f:
        for new_line in new_lines:
            f.write(new_line)