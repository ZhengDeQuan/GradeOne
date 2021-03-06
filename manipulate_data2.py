#!/usr/bin/python
# -*- coding:UTF-8 -*-
#改造 [que,ans,label]--->[que,ans,label,number_of_good_questions],并且改成了列表的列表Ques[que],que=[[que1,ans11,label,num],[que1,ans12,label,num]]
import os
import numpy as np

infiles = ['WikiQASent-test-filtered.txt','WikiQASent-dev-filtered.txt','WikiQASent-train-filtered.txt']
indir = "NewCorpus2"
outdir = "NewCorpus3"

filename = infiles[2]
infile  = os.path.join(indir,filename)
outfile = os.path.join(outdir,filename)

with open(infile,encoding= 'utf-8' ) as f:
    lines = f.readlines()
    last_que = lines[0].split('\t')[0]
    One_Que = []
    Ques = []
    for line_num , line in enumerate(lines):
        line = line.strip('\n')#当strip(rm)中的rm为空时，默认删除空白符（包括'\n', '\r',  '\t',  ' ')
        line = line.split('\t')
        #print("line = ",line)
        if last_que == line[0]:
            One_Que.append(line)
        else:
            Ques.append(One_Que)
            One_Que = []
            One_Que.append(line)
            last_que = line[0]
    Ques.append(One_Que)


for One_Que in Ques:
    One_Que.sort(key = lambda a:int(a[2]),reverse=True)

num_right_ans = 0
for One_Que in Ques:
    num_right_ans = 0
    for qa_pair in One_Que:
        if int(qa_pair[2]) == 1 :
            num_right_ans += 1
    for qa_pair in One_Que:
        qa_pair.append(str(num_right_ans))

fout = open(outfile,'w',encoding='utf-8')
for One_Que in Ques:
    for qa_pair in One_Que:
        out_content = '\t'.join(qa_pair)
        fout.write(out_content+'\n')
fout.close()

# For_return = [];
#
# for One_Que in Ques:
#     len_One_Que = len(One_Que)
#     len_right_ans = One_Que[0][3]
#     len_randint = len_One_Que - len_right_ans
#
#     for_return = []
#     for i in range(len_right_ans):
#         que = One_Que[i][0]
#         good_ans = One_Que[i][1]
#         r = np.random.randint(0,len_randint)
#         bad_ans = One_Que[len_right_ans+r][1]
#         for_return.append([que , good_ans , bad_ans])
#
#     For_return.append(for_return);
#
# For_return 就是最后要的数据集,对于每个epochs,需要重新构建这个数据集，因为需要重新random来找一个随机的bad answer
# for Q in For_return:
#     for p_3 in Q:
#         print(p_3)
#     print()