#!/usr/bin/python
# -*- coding:UTF-8 -*-
import os
import re
import gensim

model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary = True);
print("load_succeed\n")
infiles = ['WikiQASent-test-filtered.txt','WikiQASent-dev-filtered.txt','WikiQASent-train-filtered.txt']
indir = "NewCorpus4"
outdir = "NewCorpus5"

for num_of_file in range(len(infiles)):
    filename = infiles[num_of_file]
    infile = os.path.join(indir,filename)
    outfile = os.path.join(outdir,filename)
    new_lines = []
    with open(infile,'r',encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.split('\t')
            que = line[0]
            que = que.split()

            ans = line[1]
            ans = ans.split()
            for i , word in enumerate(que):
                if word not in model:
                    if word.lower() in model:
                        que[i] = word.lower()
                    elif (word.lower()[0].upper())+(word.lower()[1:]) in model:
                        que[i] = (word.lower()[0].upper())+(word.lower()[1:])
                    elif word[0].upper()+word[1:] in model:
                        que[i] = word[0].upper()+word[1:]
                    elif word.upper() in model:
                        que[i] = word.upper()

            for i, word in enumerate(ans):
                if word not in model:
                    if word.lower() in model:
                        ans[i] = word.lower()
                    elif (word.lower()[0].upper())+(word.lower()[1:]) in model:
                        ans[i] = (word.lower()[0].upper())+(word.lower()[1:])
                    elif word[0].upper()+word[1:] in model:
                        ans[i] = word[0].upper()+word[1:]
                    elif word.upper() in model:
                        ans[i] = word.upper()

            que = ' '.join(que)
            ans = ' '.join(ans)
            new_line = '\t'.join([que,ans,line[2],line[3]])
            new_lines.append(new_line)

    with open(outfile,'w',encoding='utf-8') as fout:
        for new_line in new_lines:
            fout.write(new_line)