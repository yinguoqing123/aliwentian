import imp
from urllib.parse import quote_from_bytes
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

class MyDataSet():
    def __init__(self, path, tokenizer, batch_size=32, maxlen=64, negs_num=64, harduse=False, hardnum=3):
        self.queries = self.readQuery(path)
        self.docs = self.readDoc(path)
        self.labels = self.readLabel(path)
        self.devqueries = self.readQuery(path, mode='dev')
        self.step = 0
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.permutation = np.random.permutation(len(self.labels))
        self.train_permutation = self.permutation[:-2000]
        #self.train_permutation = self.permutation
        self.tokenizer = tokenizer
        self.test_permutation = self.permutation[-2000:]
        self.negs_num = negs_num
        self.harduse = harduse
        self.hardnum = hardnum
        self.negsampels = self.getNegSamples()
        self.hardsamples = {}
        self.getHardSamples(path)
        
    def encode(self, s):
        encode_ = self.tokenizer.encode_plus(s, padding=True, truncation=True, 
                                                 max_length=self.maxlen, return_attention_mask=True)
        ids, mask = encode_['input_ids'], encode_['attention_mask']
        return torch.tensor(ids), torch.tensor(mask)
        
    def iter_permutation(self, mode='train'):
        assert mode in ('train', 'test'), 'mode is invalid'
        step = 0
        if mode == 'train':
            permutation = self.train_permutation
        if mode == 'test':
            permutation = self.test_permutation
        while True:
            start = step * self.batch_size
            end = min(len(permutation), start + self.batch_size)
            negindices = np.random.choice(self.negsampels, self.negs_num)
            queries, queries_mask, docs, docs_mask, negs, negs_mask = [], [], [], [], [], []
            queryid_hard = []
            for index in permutation[start:end]:
                queryid_hard.append(self.labels[index][0])
                query = self.queries[self.labels[index][0]]
                query, query_mask = self.encode(query)
                queries.append(query)
                queries_mask.append(query_mask)
                
                doc = self.docs[self.labels[index][1]]
                doc, mask = self.encode(doc)
                docs.append(doc)
                docs_mask.append(mask)
                
            for index in negindices:
                negdoc = self.docs[index]
                negdoc, mask = self.encode(negdoc)
                negs.append(negdoc)
                negs_mask.append(mask)
                
            queries = pad_sequence(queries, batch_first=True)
            queries_mask = pad_sequence(queries_mask, batch_first=True)
            docs = pad_sequence(docs, batch_first=True)
            docs_mask = pad_sequence(docs_mask, batch_first=True)
            negs = pad_sequence(negs, batch_first=True)
            negs_mask = pad_sequence(negs_mask, batch_first=True)

            if self.harduse:
                hards, hards_mask = self.gethardfeat(queryid_hard, self.hardnum)
                hards = pad_sequence(hards, batch_first=True)
                hards_mask = pad_sequence(hards, batch_first=True)
                yield queries, queries_mask, docs, docs_mask, negs, negs_mask, hards, hards_mask
            else:
                yield queries, queries_mask, docs, docs_mask, negs, negs_mask
            if end == len(permutation):
                step = 0
                if mode == 'train':
                    permutation = np.random.permutation(permutation)
                else:
                    break
            else:
                step += 1 
    
    def gethardfeat(self, queryids, hardnum):
        hards, hards_mask = [], []
        for id in queryids:
            hards_docid = np.random.choice(self.hardsamples[id], hardnum)
            for docid in hards_docid:
                ids, mask = self.encode(self.docs[docid])
                hards.append(ids)
                hards_mask.append(mask)
        return hards, hards_mask
                
                
    def iter_queries(self, mode='train'):
        assert mode in ('train', 'test', 'dev'), 'mode is invalid'
        step = 0
        if mode == 'train':
            queries_origin = self.queries[1:]
        if mode == 'test':
            queries_origin = []
            for index in self.test_permutation:
                queries_origin.append(self.queries[self.labels[index][0]])
        if mode == 'dev':
            queries_origin = self.devqueries
        while True:
            start = step * self.batch_size
            end = min(len(queries_origin), start + self.batch_size)
            queries, queries_mask = [], []
            for index in range(start, end):
                query = queries_origin[index]
                query, query_mask = self.encode(query)
                queries.append(query)
                queries_mask.append(query_mask)
            
            queries = pad_sequence(queries, batch_first=True)
            queries_mask = pad_sequence(queries_mask, batch_first=True)
            yield queries, queries_mask
            if end == len(queries_origin):
                step = 0
                break
            else:
                step += 1
            
    def iter_docs(self):
        step = 0
        while True:
            start = step * self.batch_size
            end = min(len(self.docs), start + self.batch_size)
            docs, docs_mask = [], []
            for index in range(start, end): 
                if index == 0:
                    continue
                doc = self.docs[index]
                doc, mask = self.encode(doc)
                docs.append(doc)
                docs_mask.append(mask)
            docs = pad_sequence(docs, batch_first=True)
            docs_mask = pad_sequence(docs_mask, batch_first=True)
            yield docs, docs_mask
            if end == len(self.docs):
                step = 0
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
    
    def readHardSmples(self, path, mode='hard'):
        hardSamples = {}
        if not os.path.exists(path + 'hard_samples.txt') or mode != 'hard':
            random_p = list(range(1, len(self.docs)))
            hard_samp = np.random.choice(random_p, self.hard_num)
            for i in range(100000):
                # hardSamples[i+1] = np.random.choice(random_p, self.hard_num)  # 执行太慢了
                hardSamples[i+1] = hard_samp
        else:
            with open(path + 'hard_samples.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    queryid, hardids = line.strip().split('\t')
                    hardSamples[int(queryid)] = [int(id) for id in hardids.split(',')]  
        self.hardsamples = hardSamples
        
    def getHardSamples(self, path, mode='hard'):
        hardSamples = {}
        if not os.path.exists(path + 'hard_samples.txt') or mode != 'hard':
            pass
        else:
            with open(path + 'hard_samples.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    queryid, hardids = line.strip().split('\t')
                    hardSamples[int(queryid)] = [int(id) for id in hardids.split(',')[500:600]]  
        self.hardsamples = hardSamples
        
    def __len__(self):
        return (len(self.train_permutation)+self.batch_size-1) // self.batch_size
            
    def getNegSamples(self):
        poses = [label[1] for label in self.labels]
        negs = list(set(range(1, 1001501))-set(poses))
        return negs
