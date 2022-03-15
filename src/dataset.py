import torch 
from torch import nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
import jieba
import gensim

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
    def __init__(self, path, word2id, batch_size=32, maxlen=32, negs_num=32):
        self.queries = self.readQuery(path)
        self.docs = self.readDoc(path)
        self.labels = self.readLabel(path)
        self.devqueries = self.readQuery(path, mode='dev')
        self.step = 0
        self.batch_size = batch_size
        self.word2id = word2id
        self.maxlen = maxlen
        self.permutation = np.random.permutation(len(self.labels))
        self.train_permutation = self.permutation[:-10000]
        self.test_permutation = self.permutation[-10000:]
        self.negs_num = negs_num
        self.negsampels = self.getNegSamples()
        
    def iter_train_permutation(self):
        while True:
            start = self.step * self.batch_size
            end = min(len(self.train_permutation), start + self.batch_size)
            negindices = np.random.choice(self.negsampels, self.negs_num)
            queries, docs, negs = [], [], []
            for index in self.train_permutation[start:end]:
                query = self.queries[self.labels[index][0]]
                #query = ''.join([' ' + char + ' ' if self.isDigitOrAlpha(char) else char for char in query])  # 数字和字母是否单独切分
                queryid = []
                for subquery in query.split():
                    subquery = jieba.lcut(subquery, HMM=False)
                    subquery = [self.word2id.get(word, 1) for word in subquery]
                    queryid.extend(subquery)
                queries.append(torch.tensor(queryid[:self.maxlen]))
                
                doc = self.docs[self.labels[index][1]]
                #doc = ''.join([' ' + char + ' ' if self.isDigitOrAlpha(char) else char for char in doc])
                docid = []
                for subdoc in doc.split():
                    subdoc = jieba.lcut(subdoc, HMM=False)
                    subdoc = [self.word2id.get(word, 1) for word in subdoc]
                    docid.extend(subdoc)
                docs.append(torch.tensor(docid[:self.maxlen]))
                
            for index in negindices:
                negdoc = self.docs[index]
                #negdoc = ''.join([' ' + char + ' ' if self.isDigitOrAlpha(char) else char for char in negdoc])
                docid = []
                for subdoc in negdoc.split():
                    subdoc = jieba.lcut(subdoc, HMM=False)
                    subdoc = [self.word2id.get(word, 1) for word in subdoc]
                    docid.extend(subdoc)
                negs.append(torch.tensor(docid[:self.maxlen]))
            
            queries = pad_sequence(queries, batch_first=True)
            docs = pad_sequence(docs, batch_first=True)
            negs = pad_sequence(negs, batch_first=True)
            yield queries, docs, negs
            if end == len(self.train_permutation):
                self.step = 0
                self.train_permutation = np.random.permutation(self.train_permutation)
            else:
                self.step += 1 
                
    def iter_test_permutation(self):
        step = 0
        while True:
            start = step * self.batch_size
            end = min(len(self.test_permutation), start + self.batch_size)
            negindices = np.random.choice(self.negsampels, self.negs_num)
            queries, docs, negs = [], [], []
            for index in self.test_permutation[start:end]:
                query = self.queries[self.labels[index][0]]
                #query = ''.join([' ' + char + ' ' if self.isDigitOrAlpha(char) else char for char in query])  # 数字和字母是否单独切分
                queryid = []
                for subquery in query.split():
                    subquery = jieba.lcut(subquery, HMM=False)
                    subquery = [self.word2id.get(word, 1) for word in subquery]
                    queryid.extend(subquery)
                queries.append(torch.tensor(queryid[:self.maxlen]))
                
                doc = self.docs[self.labels[index][1]]
                #doc = ''.join([' ' + char + ' ' if self.isDigitOrAlpha(char) else char for char in doc])
                docid = []
                for subdoc in doc.split():
                    subdoc = jieba.lcut(subdoc, HMM=False)
                    subdoc = [self.word2id.get(word, 1) for word in subdoc]
                    docid.extend(subdoc)
                docs.append(torch.tensor(docid[:self.maxlen]))
            
            for index in negindices:
                negdoc = self.docs[index]
                #negdoc = ''.join([' ' + char + ' ' if self.isDigitOrAlpha(char) else char for char in negdoc])
                docid = []
                for subdoc in negdoc.split():
                    subdoc = jieba.lcut(subdoc, HMM=False)
                    subdoc = [self.word2id.get(word, 1) for word in subdoc]
                    docid.extend(subdoc)
                negs.append(torch.tensor(docid[:self.maxlen]))
            
            queries = pad_sequence(queries, batch_first=True)
            docs = pad_sequence(docs, batch_first=True)
            negs = pad_sequence(negs, batch_first=True)
            yield queries, docs, negs
            if end == len(self.test_permutation):
                break
            else:
                step += 1
                
    def iter_dev_queries(self):
        step = 0
        while True:
            start = step * self.batch_size
            end = min(len(self.devqueries), start + self.batch_size)
            queries = []
            for index in range(start, end):
                query = self.devqueries[index]
                #query = ''.join([' ' + char + ' ' if self.isDigitOrAlpha(char) else char for char in query])  # 数字和字母是否单独切分
                queryid = []
                for subquery in query.split():
                    subquery = jieba.lcut(subquery, HMM=False)
                    subquery = [self.word2id.get(word, 1) for word in subquery]
                    queryid.extend(subquery)
                queries.append(torch.tensor(queryid[:self.maxlen]))
            
            queries = pad_sequence(queries, batch_first=True)
            yield queries
            if end == len(self.devqueries):
                break
            else:
                step += 1   
                
    def iter_docs(self):
        step = 0
        while True:
            start = step * self.batch_size
            end = min(len(self.docs), start + self.batch_size)
            docs, negs = []
            for index in range(start, end): 
                doc = self.docs[index+1]
                #doc = ''.join([' ' + char + ' ' if self.isDigitOrAlpha(char) else char for char in doc])
                docid = []
                for subdoc in doc.split():
                    subdoc = jieba.lcut(subdoc, HMM=False)
                    subdoc = [self.word2id.get(word, 1) for word in subdoc]
                    docid.extend(subdoc)
                docs.append(torch.tensor(docid[:self.maxlen]))
        
            docs = pad_sequence(docs, batch_first=True)
            yield docs
            if end == len(self.docs):
                break
            else:
                step += 1
    
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
        
    def __len__(self):
        return (len(self.train_permutation)+self.batch_size-1) // self.batch_size
    
    def isDigitOrAlpha(self, char):
        return ( ord(char) >= ord('a') and ord(char)<=ord('z') ) or ( ord(char)>=ord('A') and ord(char)<=ord('Z') ) \
            or ( ord(char)>=ord('0') and ord(char)<=ord('9') )
            
    def getNegSamples(self):
        poses = [label[1] for label in self.labels]
        negs = list(set(range(1, 1000001))-set(poses))
        return negs
