#!/usr/bin/python
# -*- coding:UTF-8 -*-

import os
import numpy as np

infiles = ['WikiQASent-test-filtered.txt','WikiQASent-dev-filtered.txt','WikiQASent-train-filtered.txt']
indir = "NewCorpus6"

sentences_len = np.zeros(118,'int32')
#只有一个244的长度的超长句子,而且是答案，而不是问题，所以可以损失这一个句子，而剪短输入句子的长度
if __name__ == "__main__":
    for num_of_files in range(3):
        filename = os.path.join(indir , infiles[num_of_files])
        with open(filename,'r',encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.split('\t')
                for i in range(2):
                    sentence = line[i].split()
                    if len(sentence) > len(sentences_len):
                        continue;
                    sentences_len[len(sentence)] += 1
    print(sentences_len)
