#!/usr/bin/python
# -*- coding:UTF-8 -*-
import os
import re
import pickle
import numpy as np

indir = "NewCorpus7"
# train_file = os.path.join(indir,'WikiQASent-train-filtered.txt')
# valid_file = os.path.join(indir,'WikiQASent-dev-filtered.txt')
# test_file = os.path.join(indir,'WikiQASent-test-filtered.txt')
train_file = 'WikiQASent-train-filtered.txt'
valid_file = 'WikiQASent-dev-filtered.txt'
test_file = 'WikiQASent-test-filtered.txt'


EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 118
hidden_lstm = 300
hidden_attention = 300
Margin = 0.05
###########
# 数据提取 #
###########
'''
先load trian 和 valid的数据，在最后的测试阶段再load test集
'''
def str_to_int(line):
    #line是一个列表['que','ans','label','num_of……']每个元素都是字符串形式的数字，这个函数要将他们转换成真正的数字
    for i in range(4):
        line[i] = line[i].split()
        for j in range(len(line[i])):
            line[i][j] = int(line[i][j])
    return line

def get_data(indir,filename):
    #相同问题的问答对放到一个list里面
    train_file = os.path.join(indir,filename)
    with open(train_file, 'r') as f:
        lines = f.readlines()
        # lines中的数据格式，que，ans，label，num_of_good_ans_to_the_question
        One_Que = []
        Ques = []
        last_que = lines[0].split('\t')[0]
        for line in lines:
            line = line.split('\t')
            if last_que == line[0]:
                # 还是同一个问题、答案对
                One_Que.append(str_to_int(line))
            else:
                # 是一个新的问题、答案对了
                Ques.append(One_Que)
                # 完成上一套问题的操作

                One_Que = []
                last_que = line[0]#在line还没有被修改成list的list之前赋值
                One_Que.append(str_to_int(line))
                # 为下一套问题的操作做好准备
        Ques.append(One_Que)
        # 将最后一套问题装入
        # for One_Que in Ques:
        #     One_Que.sort(key=lambda a: int(a[2]), reverse=True)
        # 同一套问题中，将正确答案排在前面，将错误答案排在后面
        #之前都是排过序的，不用再排序了
    return Ques


#######
# 训练 #
#######
def get_Med_data(Ques):
    #得到que ans1 ans2形式的三元组的数据
    For_return = []
    for One_Que in Ques:
        len_One_Que = len(One_Que)
        len_good_ans = int(One_Que[0][3][0])
        # print("lenQues=",len_One_Que)
        # print("lengood=",len_good_ans)
        if len_good_ans == len_One_Que :
            #对于全部答案都是正确的，这种问题，我不想浪费，就在其他的问题中，随便找一个答案作为它的bad_answer
            len_Ques = len(Ques)
            for i in range(len_One_Que):
                que = One_Que[i][0]
                good_ans = One_Que[i][1]
                r1 = np.random.randint(0,len_Ques)
                len_randint = len(Ques[r1])
                r2 = np.random.randint(0,len_randint)
                bad_ans = Ques[r1][r2][1]
                For_return.append([que,good_ans,bad_ans])

        else:
            len_randint = len_One_Que - len_good_ans
            for i in range(len_good_ans):
                que = One_Que[i][0]
                good_ans = One_Que[i][1]
                r = np.random.randint(0,len_randint)
                bad_ans = One_Que[len_good_ans + r][1]
                For_return.append([que,good_ans,bad_ans])
    return For_return

def get_Padded_data(Ques,MAX_SEQUENCE_LENGTH):
    # 现在的Ques的shape是[batch_size,3,?]，因为每个句子的长度不等，所以第三个维度不统一，需要用0 pad
    for_sen = []
    For_return = []
    for line in Ques:
        for_sen = []
        for sen in line:#对于train来说是que ,ans1 ,ans2 ,对于valid，test来说是que ，ans
            sentence = np.zeros(MAX_SEQUENCE_LENGTH,dtype='float32')
            # sentence_array = np.zeros((MAX_SEQUENCE_LENGTH,),dtype="int32")
            # sentence = list(sentence_array)
            # sentence = []
            # for i in range(MAX_SEQUENCE_LENGTH):
            #     sentence.append(int(0))
            for i ,word_id in enumerate(sen):
                if i >= MAX_SEQUENCE_LENGTH :
                    break
                sentence[i] = word_id
            for_sen.append(sentence)
        #for_sen = np.array(for_sen)
        For_return.append(for_sen)
    #For_return = np.array(For_return)
    return For_return

def get_triple(Data):
    #shape[?,3,MAX_SEQUENCE_LEN]
    np.random.shuffle(Data)
    Que = []
    Ans1 = []
    Ans2 = []
    for line in Data:
        Que.append(line[0])
        Ans1.append(line[1])
        Ans2.append(line[2])
    Que = np.array(Que)
    Ans1 = np.array(Ans1)
    Ans2 = np.array(Ans2)
    return Que,Ans1,Ans2

def make_fake_y3(length):
    return np.zeros((length,118,600),dtype ='int32')

def make_fake_y2(length):
    return np.zeros((length,1500))

def make_fake_y(length):
    return np.zeros((length,),dtype ='int32')