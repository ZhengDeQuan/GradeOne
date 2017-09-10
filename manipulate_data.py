#!/usr/bin/python
# -*- coding:UTF-8 -*-

import os

filenames = ['test.txt' , 'WikiQASent-test-filtered.txt','WikiQASent-dev-filtered.txt','WikiQASent-train.txt']
filename = filenames[3]
flag = True
question_without_good_answer = 0
with open(filename,encoding= 'utf-8' ) as f:
    lines = f.readlines()
    last_que = lines[0].split('\t')[0]
    the_question_has_right_answer = False;
    for line_num , line in enumerate(lines):
        line.strip('\n')
        line = line.split('\t')
        if last_que == line[0]:
            if int(line[2]) == 1:
                #print("right_ans_line_num = ",line_num);
                the_question_has_right_answer = True;
        else:
            #一个新的问题到了，
            #评价上一组问题是否有正确答案
            if the_question_has_right_answer == False:
                print("line_num=",line_num);
                question_without_good_answer +=1
            #处理这一组的，初始化问题
            the_question_has_right_answer = False
            last_que = line[0]

            #处理这一组的
            if int(line[2]) == 1:
                the_question_has_right_answer = True;