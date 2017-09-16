#!/usr/bin/python
# -*- coding:UTF-8 -*-

# filenames = ['test1.txt' , 'WikiQASent-test-filtered.txt','WikiQASent-dev-filtered.txt','WikiQASent-train.txt']
# filename = filenames[0]
#
# with open(filename,'a',encoding='utf-8') as f:
#     f.write("dnfjklsenklfjsdkl;\n")
#
# with open(filename,'r',encoding='utf-8') as f:
#     lines = f.readlines()
#     print(lines)

# Ques = [[[1,1,0],[1,2,1],[1,3,1],[1,4,0]],
#         [[2,1,0],[2,2,1],[2,3,1],[2,4,0],[2,5,0]]
#         ]
# New_Ques = []
# new_que = []
# for que in Ques:
#     new_que = sorted(que,key = lambda a:int(a[2]),reverse=True)
#     New_Ques.append(new_que)
# for que in New_Ques:
#     print(que)
#
# for que in Ques:
#     que.sort(key = lambda a:int(a[2]),reverse=True)
#
# for que in Ques:
#     print(que)

# a = "i love you which yesr 's 1986 ~ 1987 9 9 9      0";
# # la = a.split();
# # print(la)
# import re
# pattern = re.compile('[0-9]',re.S)
# #定义一个局部函数，就应该能搞定，将这个函数，和那个将要用到的a，也就是句子都放在同一个局部中
# def f0_9(i):
#     return ' '+a[i.start()]+' '
#
# a = re.sub(pattern , f0_9 ,a)
# print(a)
# la = a.split()
# print(la)



# import re
# import os
#
# pattern = re.compile('[0-9]',re.S)
# def myfunction(line):
#
#     def rep0_9( i ):
#         return ' '+line[i.start()]+' '
#
#     new_line = re.sub(pattern, rep0_9 , line)
#     return new_line
#
# infiles = ['WikiQASent-test-filtered.txt','WikiQASent-dev-filtered.txt','WikiQASent-train.txt',
#            'WikiQASent-dev.txt','WikiQASent-test.txt']
# dir = "NewCorpus"
# infile = infiles[4]
# outfile = os.path.join(dir,infile)
#
# with open(infile , 'r',encoding= 'utf-8') as fin:
#     in_lines = fin.readlines()
#     out_lines = []
#     for line in in_lines:
#         new_line = myfunction(line)
#         out_lines.append(new_line)
#
#     with open(outfile,'w',encoding = 'utf-8') as fout:
#         for new_line in out_lines:
#             fout.write(new_line)


#这段代码证明，修改后的labels列变成了' '+label+' '，但是仍然能用label=int(line[2])提取出整型的label
# import os
# infiles = ['WikiQASent-test-filtered.txt','WikiQASent-dev-filtered.txt','WikiQASent-train.txt']
# dir = "NewCorpus"
# infile = infiles[2]
# outfile = os.path.join(dir,infile)
# with open(outfile,'r',encoding='utf-8') as f:
#     lines = f.readlines()
#     for line in lines:
#         line = line.split('\t')
#         print(line)
#         label = int(line[2])
#         if label == 0:
#             print(type(label))
#         a = input("}{")


sdict = {}
for i in range(10):
    if i not in sdict:
        sdict[i] = []
    for j in range(10):
        sdict[i].append(j)
print(sdict)