#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import cPickle as pickle
import numpy

##############
#
# Author: Li Pingjiang
# Date: 2018.01.08
#
# This is used for read a text file contains only index-format data
# And output the embedding of it
#
#
##############

' word2vec module for general purpose, uses index as input and output '

__author__ = 'Li Pingjiang'

class CBOW(nn.Module):
    def __init__(self, n_word, n_dim, context_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(n_word, n_dim)
        self.linear1 = nn.Linear(context_size * n_dim, 128) # content_size is the x length in total
        self.linear2 = nn.Linear(128, n_word)

    def forward(self, x):  # x is the list of index of before and after words, as the format of Variable
        x = self.embedding(x)
        x = x.view(1, -1)
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        x = self.linear2(x)
        x = F.log_softmax(x)
        return x

class INDW2V:
    def __init__(self,INDEX_SIZE,EMBED_DIM,CONTEXT_SIZE,LR=1e-3):
        self.INDEX_SIZE=INDEX_SIZE
        self.EMBED_DIM=EMBED_DIM
        self.CONTEXT_SIZE=CONTEXT_SIZE
        self.model = CBOW(INDEX_SIZE, EMBED_DIM, CONTEXT_SIZE)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=LR)
        'Finish Init'
    def load(self, input_path):
        self.data = []
        with open(input_path, 'r') as f:
            for line in f.readlines():
                # print(line.strip())
                context = map( int , line.strip().split(" ")[0].split(",") )
                target = int( line.strip().split(" ")[1] )
                if( len(context) ==  self.CONTEXT_SIZE):
                    self.data.append((context, target))
                # print(str(context) + " " + target)
        print 'Finish Loading Data: ' + str( len(self.data) )
    def trian(self,EPOCHS=10):
        data=self.data
        model=self.model
        criterion=self.criterion
        optimizer=self.optimizer
        for epoch in range(EPOCHS):
            print('*' * 10)
            print('epoch {}'.format(epoch))
            running_loss = 0
            for word in data:
                context, target = word
                # print("content",context)
                # print("target",target)
                context = Variable(
                    torch.LongTensor(context))
                target = Variable(torch.LongTensor([target]))
                # print("target", target)
                if torch.cuda.is_available():
                    context = context.cuda()
                    target = target.cuda()
                # forward
                out = model(context)
                # print("out", out)
                # print("target", target)
                loss = criterion(out, target)  # out is a list of prosibility of each words, target is the index of correct one
                running_loss += loss.data[0]
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('loss: {:.6f}'.format(running_loss / len(data)))

    def save_object(self,OUTPUT_PATH):
        pickle.dump(self.model.embedding, open(OUTPUT_PATH, "w"), True)  # compress

class DataConvertor:
    def __init__(self):
        self.data = []
        'Finish'
    # CONTENT_LENTH is the total length include the words before and after the target
    def raw2input(self,CONTENT_LENTH=4,raw_path="data/mil",input_path="data/index_input"):
        self.data = []
        with open(raw_path, 'r') as f:
            for line in f:
                # print line.strip().split("\t")
                if(line.strip()!=''):
                    line = map( int, line.split("\t"))
                    # print(line)
                    for i in range( CONTENT_LENTH/2 , len(line)- CONTENT_LENTH/2):
                        context=[]
                        target=line[i]
                        for j in range(1,CONTENT_LENTH/2+1):
                            context.append(line[i+j])
                            context.append(line[i-j])
                        self.data.append((context, target))
                    # print(self.data)
                    # print("Finish: " + str(line))
            # exit()
    def save_input(self,output_path="data/index_input2"):
        with open(output_path, 'w') as file:
            for element in self.data:
                # print(element)
                # print(element[0])
                # print ( ','.join( map(str,element[0])) + "\t" + str(element[1]) )
                file.write(','.join( map(str,element[0])) + "\t" + str(element[1]) + "\n")


def test():
    dc = DataConvertor()
    dc.raw2input()
    dc.save_input()
    print("Finish Test")
if __name__=='__main__':
    test()
    # embedder = INDW2V(INDEX_SIZE=8, EMBED_DIM=100, CONTEXT_SIZE=4)
    # embedder.load('data/index_input')
    # embedder.trian(EPOCHS=10)
    # embedder.save_object(OUTPUT_PATH="data/embedding.pickle")

