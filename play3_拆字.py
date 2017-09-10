#!/usr/bin/python
#-*- coding:UTF-8 -*-
import os
import re
import gensim

model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary = True);
print("load_succeed\n")
infiles = ['WikiQASent-test-filtered.txt','WikiQASent-dev-filtered.txt','WikiQASent-train-filtered.txt']
indir = "NewCorpus5"
outdir = "NewCorpus6"



def Judge(word):
    #return [true/false, formatted word]
    if word in model:
        return [True , word]
    else:
        if word.lower() in model:
            return [True ,word.lower()]
        elif (word.lower()[0].upper()) + (word.lower()[1:]) in model:
            return [True ,(word.lower()[0].upper()) + (word.lower()[1:])]
        elif word[0].upper() + word[1:] in model:
            return [True ,word[0].upper() + word[1:]]
        elif word.upper() in model:
            return [True , word.upper()]
        else :
            return [False, word]

if __name__ == "__main__":
    for num_of_file in range(len(infiles)):
        filename = infiles[num_of_file]
        infile = os.path.join(indir, filename)
        outfile = os.path.join(outdir, filename)
        new_lines = []
        with open(infile, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.split('\t')
                que = line[0]
                que = que.split()
                ans = line[1]
                ans = ans.split()
                # print("que = ",que)
                # print("ans = ",ans)
                new_que = []
                new_ans = []
                for i, word in enumerate(que):
                    if word in model:
                        new_que.append(word)
                    else:
                        # split the word into two word
                        if len(word) == 1:
                            new_que.append(word)
                        else:
                            flag_get_good = False
                            for split_iter in range(1, len(word)):
                                front = word[:split_iter]
                                back = word[split_iter:]
                                Front = Judge(front)
                                Back = Judge(back)
                                if (Front[0] == True) and (Back[0] == True):
                                    new_que.append(Front[1])
                                    new_que.append(Back[1])
                                    print("Front = ", Front[1])
                                    print("Back = ", Back[1])
                                    flag_get_good = True
                                    break

                            if flag_get_good == False:
                                new_que.append(word)

                for i, word in enumerate(ans):
                    if word in model:
                        new_ans.append(word)
                    else:
                        # split the word into two word
                        if len(word) == 1:
                            new_ans.append(word)
                        else:
                            flag_get_good = False
                            for split_iter in range(1, len(word)):
                                front = word[:split_iter]
                                back = word[split_iter:]
                                Front = Judge(front)
                                Back = Judge(back)
                                if (Front[0] == True) and (Back[0] == True):
                                    print("F = ", Front[1])
                                    print("B = ", Back[1])
                                    new_ans.append(Front[1])
                                    new_ans.append(Back[1])
                                    flag_get_good = True
                                    break

                            if flag_get_good == False:
                                new_ans.append(word)

                new_que = ' '.join(new_que)
                new_ans = ' '.join(new_ans)
                new_line = '\t'.join([new_que, new_ans, line[2], line[3]])
                new_lines.append(new_line)

        with open(outfile, 'w', encoding='utf-8') as fout:
            for new_line in new_lines:
                fout.write(new_line)