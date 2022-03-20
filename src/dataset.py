from numpy import pad
import torch 
from torch import nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
import jieba
import gensim
import os

jieba.initialize()
path = r'D:\ai-risk\aliwentian\word2vec\word2vec.bin'
path = '../word2vec/word2vec.bin'
word2vec = gensim.models.KeyedVectors.load(path)

#取腾讯词向量的前100万个
for word in  word2vec.index_to_key[:1000000]:
    if word not in jieba.dt.FREQ:
        jieba.add_word(word)
        
#id2word = {i+2:j  for i, j in enumerate(word2vec.index_to_key[:1000000])}  # 0: <pad> 1:<unk>
#word2id = {j: i for i, j in id2word.items()}

# word2vec = word2vec.vectors[:1000000]
# word_dim = word2vec.shape[1]
# word2vec = np.concatenate([np.zeros((2, word_dim)), word2vec])

class MyDataSet():
    def __init__(self, path, word2id, char2id, batch_size=32, maxlen=64, negs_num=64, hard_num=5):
        self.queries = self.readQuery(path)
        self.docs = self.readDoc(path)
        self.labels = self.readLabel(path)
        self.devqueries = self.readQuery(path, mode='dev')
        self.step = 0
        self.batch_size = batch_size
        self.word2id = word2id
        self.char2id = char2id
        self.maxlen = maxlen
        self.permutation = np.random.permutation(len(self.labels))
        self.train_permutation = self.permutation[:-2000]
        #self.train_permutation = self.permutation
        self.test_permutation = self.permutation[-2000:]
        self.negs_num = negs_num
        self.hard_num = hard_num
        self.negsampels = self.getNegSamples()
        self.hardsamples = {}
        self.readHardSmples(path)
        
    def iter_train_permutation(self):
        while True:
            start = self.step * self.batch_size
            end = min(len(self.train_permutation), start + self.batch_size)
            negindices = np.random.choice(self.negsampels, self.negs_num)
            queries, docs, negs = [], [], []
            queries_char, docs_char, negs_char = [], [], []
            hardids = []
            for index in self.train_permutation[start:end]:
                query = self.queries[self.labels[index][0]]
                query = ''.join([' ' + char + ' ' if self.isDigitOrAlpha(char) else char for char in query])  # 数字和字母是否单独切分
                queryid, queryidchar = [], []
                for subquery in query.split():
                    subquerychar = [self.char2id.get(char, 1) for char in subquery]
                    subquery = jieba.lcut(subquery, HMM=False)
                    subquery = self.words2charid(subquery)
                    queryid.extend(subquery)
                    queryidchar.extend(subquerychar)
                queries.append(torch.tensor(queryid[:self.maxlen]))
                queries_char.append(torch.tensor(queryidchar[:self.maxlen]))
                
                doc = self.docs[self.labels[index][1]]
                doc = ''.join([' ' + char + ' ' if self.isDigitOrAlpha(char) else char for char in doc])
                docid, docidchar = [], []
                for subdoc in doc.split():
                    docchar = [self.char2id.get(char, 1) for char in subdoc]
                    subdoc = jieba.lcut(subdoc, HMM=False)
                    subdoc = self.words2charid(subdoc)
                    docid.extend(subdoc)
                    docidchar.extend(docchar)
                    
                docs.append(torch.tensor(docid[:self.maxlen]))
                docs_char.append(torch.tensor(docidchar[:self.maxlen]))
                
                if len(self.hardsamples) > 0:
                    hard_docid = self.hardsamples[self.labels[index][0]] 
                    try:
                        hard_docid.remove(self.queries[self.labels[index][1]])
                    except:
                        pass
                    hard_index = np.random.choice(hard_docid, self.hard_num)
                else:
                    hard_index = []
                    
                hardids.append(hard_index)
            
            hardnegs, hardnegs_char = [], []
            for index in range(len(hardids)):
                for id in hardids[index]:
                    doc = self.docs[id]
                    doc = ''.join([' ' + char + ' ' if self.isDigitOrAlpha(char) else char for char in doc])
                    docid, docidchar = [], []
                    for subdoc in doc.split():
                        docchar = [self.char2id.get(char, 1) for char in subdoc]
                        subdoc = jieba.lcut(subdoc, HMM=False)
                        subdoc = self.words2charid(subdoc)
                        docid.extend(subdoc)
                        docidchar.extend(docchar)
                    
                    hardnegs.append(torch.tensor(docid[:self.maxlen]))
                    hardnegs_char.append(torch.tensor(docidchar[:self.maxlen]))
                    
            hardnegs = pad_sequence(hardnegs, batch_first=True)
            hardnegs_char = pad_sequence(hardnegs_char, batch_first=True)  
            hardnegs = hardnegs.reshape(len(hardids), self.hard_num, -1)
            hardnegs_char = hardnegs_char.reshape(len(hardids), self.hard_num, -1)

            for index in negindices:
                negdoc = self.docs[index]
                negdoc = ''.join([' ' + char + ' ' if self.isDigitOrAlpha(char) else char for char in negdoc])
                docid, docidchar = [], []
                for subdoc in negdoc.split():
                    docchar = [self.char2id.get(char, 1) for char in subdoc]
                    subdoc = jieba.lcut(subdoc, HMM=False)
                    subdoc = self.words2charid(subdoc)
                    docid.extend(subdoc)
                    docidchar.extend(docchar)
                negs.append(torch.tensor(docid[:self.maxlen]))
                negs_char.append(torch.tensor(docidchar[:self.maxlen]))
            
            queries = pad_sequence(queries, batch_first=True)
            queries_char = pad_sequence(queries_char, batch_first=True)
            docs = pad_sequence(docs, batch_first=True)
            docs_char = pad_sequence(docs_char, batch_first=True)
            negs = pad_sequence(negs, batch_first=True)
            negs_char = pad_sequence(negs_char, batch_first=True)
            yield queries, queries_char, docs, docs_char, negs, negs_char, hardnegs, hardnegs_char
            if end == len(self.train_permutation):
                self.step = 0
                self.train_permutation = np.random.permutation(self.train_permutation)
            else:
                self.step += 1 
                
    def get_query_tokens(self, query):
        queries, queries_char = [], []
        query = ''.join([' ' + char + ' ' if self.isDigitOrAlpha(char) else char for char in query])  # 数字和字母是否单独切分
        queryid, queryidchar = [], []
        for subquery in query.split():
            subquerychar = [self.char2id.get(char, 1) for char in subquery]
            subquery = jieba.lcut(subquery, HMM=False)
            subquery = self.words2charid(subquery)
            queryid.extend(subquery)
            queryidchar.extend(subquerychar)
        queries.append(torch.tensor(queryid[:self.maxlen]))
        queries_char.append(torch.tensor(queryidchar[:self.maxlen]))
        return queries, queries_char
                
    def iter_test_permutation(self):
        step = 0
        #test_permutation = np.random.choice(self.permutation, 10000)  #训练集中随机选取1/10
        test_permutation = self.test_permutation
        while True:
            start = step * self.batch_size
            end = min(len(test_permutation), start + self.batch_size)
            negindices = np.random.choice(self.negsampels, self.negs_num)
            queries, docs, negs = [], [], []
            queries_char, docs_char, negs_char = [], [], []
            hardids = []
            for index in test_permutation[start:end]:
                query = self.queries[self.labels[index][0]]
                query = ''.join([' ' + char + ' ' if self.isDigitOrAlpha(char) else char for char in query])  # 数字和字母是否单独切分
                queryid, queryidchar = [], []
                for subquery in query.split():
                    subquerychar = [self.char2id.get(char, 1) for char in subquery]
                    subquery = jieba.lcut(subquery, HMM=False)
                    subquery = self.words2charid(subquery)
                    queryid.extend(subquery)
                    queryidchar.extend(subquerychar)
                queries.append(torch.tensor(queryid[:self.maxlen]))
                queries_char.append(torch.tensor(queryidchar[:self.maxlen]))
                
                doc = self.docs[self.labels[index][1]]
                doc = ''.join([' ' + char + ' ' if self.isDigitOrAlpha(char) else char for char in doc])
                docid, docidchar = [], []
                for subdoc in doc.split():
                    docchar = [self.char2id.get(char, 1) for char in subdoc]
                    subdoc = jieba.lcut(subdoc, HMM=False)
                    subdoc = self.words2charid(subdoc)
                    docid.extend(subdoc)
                    docidchar.extend(docchar)
                    
                docs.append(torch.tensor(docid[:self.maxlen]))
                docs_char.append(torch.tensor(docidchar[:self.maxlen]))
            
                docs.append(torch.tensor(docid[:self.maxlen]))
                docs_char.append(torch.tensor(docidchar[:self.maxlen]))
                
                if len(self.hardsamples) > 0:
                    hard_docid = self.hardsamples[self.labels[index][0]] 
                    try:
                        hard_docid.remove(self.queries[self.labels[index][1]])
                    except:
                        pass
                    hard_index = np.random.choice(hard_docid, self.hard_num)
                else:
                    hard_index = []
                    
                hardids.append(hard_index)
                
            hardnegs, hardnegs_char = [], []
            for index in range(len(hardids)):
                for id in hardids[index]:
                    doc = self.docs[id]
                    doc = ''.join([' ' + char + ' ' if self.isDigitOrAlpha(char) else char for char in doc])
                    docid, docidchar = [], []
                    for subdoc in doc.split():
                        docchar = [self.char2id.get(char, 1) for char in subdoc]
                        subdoc = jieba.lcut(subdoc, HMM=False)
                        subdoc = self.words2charid(subdoc)
                        docid.extend(subdoc)
                        docidchar.extend(docchar)
                    
                    hardnegs.append(torch.tensor(docid[:self.maxlen]))
                    hardnegs_char.append(torch.tensor(docidchar[:self.maxlen]))
                    
            hardnegs = pad_sequence(hardnegs, batch_first=True)
            hardnegs_char = pad_sequence(hardnegs_char, batch_first=True)
            hardnegs = hardnegs.reshape(len(hardids), self.hard_num, -1)
            hardnegs_char = hardnegs_char.reshape(len(hardids), self.hard_num, -1)

            for index in negindices:
                negdoc = self.docs[index]
                negdoc = ''.join([' ' + char + ' ' if self.isDigitOrAlpha(char) else char for char in negdoc])
                docid, docidchar = [], []
                for subdoc in negdoc.split():
                    docchar = [self.char2id.get(char, 1) for char in subdoc]
                    subdoc = jieba.lcut(subdoc, HMM=False)
                    subdoc = self.words2charid(subdoc)
                    docid.extend(subdoc)
                    docidchar.extend(docchar)
                negs.append(torch.tensor(docid[:self.maxlen]))
                negs_char.append(torch.tensor(docidchar[:self.maxlen]))
            
            queries = pad_sequence(queries, batch_first=True)
            queries_char = pad_sequence(queries_char, batch_first=True)
            docs = pad_sequence(docs, batch_first=True)
            docs_char = pad_sequence(docs_char, batch_first=True)
            negs = pad_sequence(negs, batch_first=True)
            negs_char = pad_sequence(negs_char, batch_first=True)
            yield queries, queries_char, docs, docs_char, negs, negs_char, hardnegs, hardnegs_char
            if end == len(test_permutation):
                break
            else:
                step += 1
                
    def iter_train_queries(self):
        step = 0
        while True:
            start = step * self.batch_size
            end = min(len(self.queries), start + self.batch_size)
            queries, queries_char = [], []
            for index in range(start, end):
                if index == 0:
                    continue
                query = self.queries[index]
                query = ''.join([' ' + char + ' ' if self.isDigitOrAlpha(char) else char for char in query])  # 数字和字母是否单独切分
                queryid, queryidchar = [], []
                for subquery in query.split():
                    subquerychar = [self.char2id.get(char, 1) for char in subquery]
                    subquery = jieba.lcut(subquery, HMM=False)
                    subquery = self.words2charid(subquery)
                    queryid.extend(subquery)
                    queryidchar.extend(subquerychar)
                queries.append(torch.tensor(queryid[:self.maxlen]))
                queries_char.append(torch.tensor(queryidchar[:self.maxlen]))
            
            queries = pad_sequence(queries, batch_first=True)
            queries_char = pad_sequence(queries_char, batch_first=True)
            yield queries, queries_char
            if end == len(self.queries):
                break
            else:
                step += 1 
                
    def iter_test_queries(self):
        step = 0
        #test_permutation = np.random.choice(self.permutation, 10000)  #训练集中随机选取1/10
        test_permutation = self.test_permutation
        while True:
            start = step * self.batch_size
            end = min(len(test_permutation), start + self.batch_size)
            queries, queries_char = [], []
            for index in test_permutation[start:end]:
                query = self.queries[self.labels[index][0]]
                query = ''.join([' ' + char + ' ' if self.isDigitOrAlpha(char) else char for char in query])  # 数字和字母是否单独切分
                queryid, queryidchar = [], []
                for subquery in query.split():
                    subquerychar = [self.char2id.get(char, 1) for char in subquery]
                    subquery = jieba.lcut(subquery, HMM=False)
                    subquery = self.words2charid(subquery)
                    queryid.extend(subquery)
                    queryidchar.extend(subquerychar)
                queries.append(torch.tensor(queryid[:self.maxlen]))
                queries_char.append(torch.tensor(queryidchar[:self.maxlen]))
            
            queries = pad_sequence(queries, batch_first=True)
            queries_char = pad_sequence(queries_char, batch_first=True)
            yield queries, queries_char
            if end == len(test_permutation):
                break
            else:
                step += 1
                
    def iter_dev_queries(self):
        step = 0
        while True:
            start = step * self.batch_size
            end = min(len(self.devqueries), start + self.batch_size)
            queries, queries_char = [], []
            for index in range(start, end):
                query = self.devqueries[index]
                query = ''.join([' ' + char + ' ' if self.isDigitOrAlpha(char) else char for char in query])  # 数字和字母是否单独切分
                queryid, queryidchar = [], []
                for subquery in query.split():
                    subquerychar = [self.char2id.get(char, 1) for char in subquery]
                    subquery = jieba.lcut(subquery, HMM=False)
                    subquery = self.words2charid(subquery)
                    queryid.extend(subquery)
                    queryidchar.extend(subquerychar)
                queries.append(torch.tensor(queryid[:self.maxlen]))
                queries_char.append(torch.tensor(queryidchar[:self.maxlen]))
            
            queries = pad_sequence(queries, batch_first=True)
            queries_char = pad_sequence(queries_char, batch_first=True)
            yield queries, queries_char
            if end == len(self.devqueries):
                break
            else:
                step += 1 
                
    def iter_docs(self):
        step = 0
        while True:
            start = step * self.batch_size
            end = min(len(self.docs), start + self.batch_size)
            docs, docs_char = [], []
            for index in range(start, end): 
                if index == 0:
                    continue
                doc = self.docs[index]
                doc = ''.join([' ' + char + ' ' if self.isDigitOrAlpha(char) else char for char in doc])
                docid, docidchar = [], []
                for subdoc in doc.split():
                    docchar = [self.char2id.get(char, 1) for char in subdoc]
                    subdoc = jieba.lcut(subdoc, HMM=False)
                    subdoc = self.words2charid(subdoc)
                    docid.extend(subdoc)
                    docidchar.extend(docchar)
                    
                docs.append(torch.tensor(docid[:self.maxlen]))
                docs_char.append(torch.tensor(docidchar[:self.maxlen]))
        
            docs = pad_sequence(docs, batch_first=True)
            docs_char = pad_sequence(docs_char, batch_first=True)
            yield docs, docs_char
            if end == len(self.docs):
                break
            else:
                step += 1
                
    def words2charid(self, words): 
        ids = []
        for word in words:
            ids.extend([self.word2id.get(word, 1)] * len(word) )
        return ids
    
    def readQuery(self, path, mode='train'):
        if mode == 'train':
            with open(path+'train.query.txt', encoding='utf-8') as f:
                queries = [''] + [line.strip().split('\t')[1] for line in f.readlines()]
        if mode == 'dev':
            with open(path+'dev.query.txt', encoding='utf-8') as f:
                queries = [line.strip().split('\t')[1] for line in f.readlines()]
        return queries
    
    def readDoc(self, path):
        with open(path + 'corpus.tsv', encoding='utf-8') as f:
            docs = [''] + [line.strip().split('\t')[1] for line in f.readlines()]
        return docs
        
    def readLabel(self, path):
        with open(path + 'qrels.train.tsv', encoding='utf-8') as f:
            labels = [[int(id) for id in line.strip().split('\t')] for line in f.readlines()]
        return labels
    
    def readHardSmples(self, path, mode='hard'):
        hardSamples = {}
        if not os.path.exists(path + 'hard_samples.txt') or mode != 'hard':
            random_p = list(range(1, len(self.docs)))
            for i in range(len(self.labels)):
                hardSamples[i+1] = np.random.choice(random_p, self.hard_num)
        else:
            with open(path + 'hard_samples.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    queryid, hardids = line.strip().split('\t')
                    hardSamples[int(queryid)] = [int(id) for id in hardids.split(',')]  
        self.hardsamples = hardSamples
        
    def __len__(self):
        return (len(self.train_permutation)+self.batch_size-1) // self.batch_size
    
    def isDigitOrAlpha(self, char):
        return ( ord(char) >= ord('a') and ord(char)<=ord('z') ) or ( ord(char)>=ord('A') and ord(char)<=ord('Z') ) \
            or ( ord(char)>=ord('0') and ord(char)<=ord('9') )
            
    def getNegSamples(self):
        poses = [label[1] for label in self.labels]
        negs = list(set(range(1, 1001501))-set(poses))
        return negs
