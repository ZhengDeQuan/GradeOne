#!/usr/bin/python
# -*- coding:UTF-8

import os
import re

infiles = ['WikiQASent-test-filtered.txt','WikiQASent-dev-filtered.txt','WikiQASent-train-filtered.txt']
indir = "NewCorpus6"
outdir = "NewCorpus6"
innerdir = "good"

dic_unknow_word = {}
def get_unknow_word(filename):
    with open(filename,'r',encoding = 'utf-8') as fin:
        words = fin.readlines()
        for word in words:
            word = word.strip()
            dic_unknow_word[word] = 0
    return dic_unknow_word

if __name__=="__main__":
    dic_unknow_word = get_unknow_word(os.path.join(indir,'word_not_in_model.txt'))
    for num_of_file in range(len(infiles)):
        filename = os.path.join(indir,infiles[num_of_file])
        with open(filename,'r',encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.split('\t')
                for i in range(2):
                    sentence = line[i].split()
                    for word in sentence :
                        if word in dic_unknow_word:
                            dic_unknow_word[word] +=1

    # for word , value in dic_unknow_word.items():
    #     print(word ,"  ", value)
    dic_new = sorted(dic_unknow_word.items(),key = lambda item:(-item[1],item[0]))
    for word , value in dic_new:
        print(word," ",value)
    outfile = os.path.join(outdir,"unknow_word.txt")
    with open(outfile,'w',encoding='utf-8') as fout:
        for tu in dic_new:
            out_content = tu[0]+' '+str(tu[1])+'\n'
            fout.write(out_content)
