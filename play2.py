#!/usr/bin/python
# -*- coding:UTF-8 -*-

import os
import re

#将train中的没有正确答案的问题去除
infiles = ['WikiQASent-test-filtered.txt','WikiQASent-dev-filtered.txt','WikiQASent-train.txt',
           'WikiQASent-dev.txt','WikiQASent-test.txt']
dir = "NewCorpus"
infile = infiles[2]
outfile = os.path.join(dir,infile)
One_Que = []
Ques = []
with open(outfile,encoding='utf-8') as f:
    lines = f.readlines()
    last_que = lines[0].split('\t')[0]
    for line_num , line in enumerate(lines):
        line = line.strip()
        line = line.split('\t')
        if last_que == line[0]:
            One_Que.append(line)
        else:
            Ques.append(One_Que)
            One_Que = []
            One_Que.append(line)
            last_que = line[0]
    Ques.append(One_Que)

print("original line = ",len(Ques))
New_Ques = []
for One_Que in Ques:

    has_good_answer = False
    for item in One_Que:
        if int(item[2]) == 1:
            has_good_answer = True

    if has_good_answer == True:
        New_Ques.append(One_Que)

outfile2 = os.path.join('NewCorpus2',infile)
with open(outfile2,'w',encoding='utf-8') as fout:
    for One_Que in New_Ques:
        for item in One_Que:
            #out_content = item[0] +'\t'+ item[1] +'\t'+ item[2]
            out_content = '\t'.join(item)
            fout.write(out_content+'\n')
            # print(One_Que)
            # print('out_content = ',out_content)
            # a = input('uyiuo')

print("Now lines = ",len(New_Ques))