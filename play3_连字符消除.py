#!/usr/bin/python
# -*- coding:UTF-8 -*-

import os
import re

infiles = ['WikiQASent-test-filtered.txt','WikiQASent-dev-filtered.txt','WikiQASent-train-filtered.txt']
indir = "NewCorpus3"
outdir = "NewCorpus4"

for num_of_file in range(len(infiles)):
    filename = infiles[num_of_file]
    infile = os.path.join(indir,filename)
    outfile = os.path.join(outdir,filename)
    pattern = re.compile('-|–|−|_|—|­|/|,|\.|\?|’|:|⁄|;|!|“|\(|\)|ˈ|′′|{|}|”|"|′|<|>|\[|‘|\]')
    new_lines = []
    with open(infile,'r',encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            new_line = re.sub(pattern," ",line)
            new_line = re.sub("°"," ° ",new_line)
            new_line = re.sub("`"," ",new_line)
            new_line = re.sub("\+"," + ",new_line)
            new_line = re.sub("\*"," * ",new_line)
            new_line = re.sub("'"," ",new_line)
            new_line = re.sub("’"," ",new_line)
            new_line = re.sub("interloan"," inter loan ",new_line)
            new_line = re.sub("I’m"," I am ",new_line)
            new_line = re.sub("centres"," centers ",new_line)
            new_line = re.sub("Fidfaddy"," Fid faddy ",new_line)
            new_line = re.sub("qualilfying","qualifying",new_line)
            new_line = re.sub(r"\\", " ", new_line)
            new_line = re.sub(r"\|", " ", new_line)

            new_lines.append(new_line)

    with open(outfile,'w',encoding='utf-8') as fout:
        for new_line in new_lines:
            fout.write(new_line)



