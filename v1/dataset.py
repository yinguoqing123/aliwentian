from importlib_metadata import PathDistribution
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
    def __init__(self, path, word2id, char2id, tokenizer=None, batch_size=32, maxlen=64, negs_num=64, hard_num=3, hardneguse=False):
        self.queries = self.readQuery(path)
        self.docs = self.readDoc(path)
        self.labels = self.readLabel(path)
        self.devqueries = self.readQuery(path, mode='dev')
        self.tokenizer = tokenizer
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
        self.hardneguse = hardneguse
        self.hardsamples = {}
        self.readHardSamples(path)
        
    def sethardMode(self, mode):
        self.hardneguse = mode
        
    def encode(self, s):
        s = ''.join([' ' + char + ' ' if self.isDigitOrAlpha(char) else char for char in s])
        id, idchar = [], []
        for sub in s.split():
            subchar = [self.char2id.get(char, 1) for char in sub]
            sub = jieba.lcut(sub, HMM=False)
            sub = self.words2charid(sub)
            id.extend(sub)
            idchar.extend(subchar)
        return torch.tensor(id[:self.maxlen]), torch.tensor(idchar[:self.maxlen])
    
    def encode2(self, s):
        encode_ = self.tokenizer.encode_plus(s, padding=True, truncation=True, 
                                                 max_length=self.maxlen, return_attention_mask=True)
        ids, mask = encode_['input_ids'], encode_['attention_mask']
        return torch.tensor(ids), torch.tensor(mask)
    
    def iter_permutation(self, mode='train'):
        assert mode in ('train', 'test'), 'mode is in valid'
        step = 0
        if mode == 'train':
            permutation = np.random.permutation(self.train_permutation)
        if mode == 'test':
            permutation = self.test_permutation
        while True:
            start = step * self.batch_size
            end = min(len(permutation), start + self.batch_size)
            negindices = np.random.choice(self.negsampels, self.negs_num)
            queries, docs, negs = [], [], []
            queries_char, docs_char, negs_char = [], [], []
            queries_bert, docs_bert, negs_bert = [], [], []
            queries_bertmask, docs_bertmask, negs_bertmask = [], [], []
            hardids = []
            for index in permutation[start:end]:
                query = self.queries[self.labels[index][0]]
                queryid, queryidchar = self.encode(query)
                queries.append(queryid)
                queries_char.append(queryidchar)

                doc = self.docs[self.labels[index][1]]
                docid, docidchar = self.encode(doc)            
                docs.append(docid)
                docs_char.append(docidchar)
                
                queries_ids, queries_mask = self.encode2(query)
                doc_ids, doc_mask = self.encode2(doc)
                queries_bert.append(queries_ids)
                queries_bertmask.append(queries_mask)
                docs_bert.append(doc_ids)
                docs_bertmask.append(doc_mask)
                
                if self.hardneguse:
                    hard_docid = self.hardsamples[self.labels[index][0]] 
                    try:
                        hard_docid.remove(self.queries[self.labels[index][1]])
                    except:
                        pass
                    hard_index = np.random.choice(hard_docid, self.hard_num)
                else:
                    hard_index = []
                    
                hardids.append(hard_index)
                
            for index in negindices:
                negdoc = self.docs[index]
                docid, docidchar = self.encode(negdoc) 
                negs.append(docid[:self.maxlen])
                negs_char.append(docidchar[:self.maxlen])
                
                negs_ids, negs_mask = self.encode2(negdoc)
                negs_bert.append(negs_ids)
                negs_bertmask.append(negs_mask)
                
            if self.hardneguse:
                hardnegs, hardnegs_char = [], []
                hard_bert, hard_bertmask = [], []
                for index in range(len(hardids)):
                    for id in hardids[index]:
                        doc = self.docs[id]
                        docid, docidchar = self.encode(doc)
                        hardnegs.append(docid)
                        hardnegs_char.append(docidchar)
                        doc_ids, doc_mask = self.encode2(doc)
                        hard_bert.append(doc_ids)
                        hard_bertmask.append(hard_bertmask)
                        
                hardnegs = pad_sequence(hardnegs, batch_first=True)
                hardnegs_char = pad_sequence(hardnegs_char, batch_first=True)  
                hardnegs = hardnegs.reshape(len(hardids), self.hard_num, -1)
                hardnegs_char = hardnegs_char.reshape(len(hardids), self.hard_num, -1)
                hard_bert = pad_sequence(hard_bert, batch_first=True)
                hard_bertmask = pad_sequence(hard_bertmask, batch_first=True)
                hard_bert = hard_bert.reshape(len(hardids), self.hard_num, -1)
                hard_bertmask = hard_bertmask.reshape(len(hardids), self.hard_num, -1)
            
            queries = pad_sequence(queries, batch_first=True)
            queries_char = pad_sequence(queries_char, batch_first=True)
            docs = pad_sequence(docs, batch_first=True)
            docs_char = pad_sequence(docs_char, batch_first=True)
            negs = pad_sequence(negs, batch_first=True)
            negs_char = pad_sequence(negs_char, batch_first=True)
            
            queries_bert = pad_sequence(queries_bert, batch_first=True)
            queries_bertmask = pad_sequence(queries_bertmask, batch_first=True)
            docs_bert = pad_sequence(docs_bert, batch_first=True)
            docs_bertmask = pad_sequence(docs_bertmask, batch_first=True)
            negs_bert = pad_sequence(negs_bert, batch_first=True)
            negs_bertmask = pad_sequence(negs_bertmask, batch_first=True)
            
            if self.hardneguse:
                yield queries, queries_char, docs, docs_char, negs, negs_char, queries_bert, docs_bert, negs_bert, queries_bertmask, docs_bertmask, negs_bertmask, hardnegs, hardnegs_char, hard_bert, hard_bertmask
            else:
                yield queries, queries_char, docs, docs_char, negs, negs_char, queries_bert, docs_bert, negs_bert, queries_bertmask, docs_bertmask, negs_bertmask
            if end == len(permutation):
                step = 0
                if mode == 'test':
                    break
            else:
                step += 1 
     
    def iter_queries(self, mode='train'):
        step = 0
        assert mode in ('train', 'test', 'dev'),   "mode is invalid"
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
            queries, queries_char = [], []
            queries_bert, queries_bertmask = [], []
            for index in range(start, end):
                query = queries_origin[index]
                queryid, queryidchar = self.encode(query)
                queries.append(queryid)
                queries_char.append(queryidchar)
                
                bert_ids, bert_mask = self.encode2(query)
                queries_bert.append(bert_ids)
                queries_bertmask.append(bert_mask)
                
            
            queries = pad_sequence(queries, batch_first=True)
            queries_char = pad_sequence(queries_char, batch_first=True)
            
            queries_bert = pad_sequence(queries_bert, batch_first=True)
            queries_bertmask = pad_sequence(queries_bertmask, batch_first=True)
            
            yield queries, queries_char, queries_bert, queries_bertmask
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
            docs, docs_char = [], []
            docs_bert, docs_bertmask = [], []
            for index in range(start, end): 
                if index == 0:
                    continue
                doc = self.docs[index]
                docid, docidchar = self.encode(doc)
                docs.append(docid)
                docs_char.append(docidchar)
                
                doc_ids, doc_mask = self.encode2(doc)
                docs_bert.append(doc_ids)
                docs_bertmask.append(doc_mask)
                
            docs = pad_sequence(docs, batch_first=True)
            docs_char = pad_sequence(docs_char, batch_first=True)
            docs_bert = pad_sequence(docs_bert, batch_first=True)
            docs_bertmask = pad_sequence(docs_bertmask, batch_first=True)
            yield docs, docs_char, docs_bert, docs_bertmask
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
    
    def readHardSamples(self, path, mode='hard'):
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
    
    def isDigitOrAlpha(self, char):
        return ( ord(char) >= ord('a') and ord(char)<=ord('z') ) or ( ord(char)>=ord('A') and ord(char)<=ord('Z') ) \
            or ( ord(char)>=ord('0') and ord(char)<=ord('9') )
            
    def getNegSamples(self):
        poses = [label[1] for label in self.labels]
        negs = list(set(range(1, 1001501))-set(poses))
        return negs
