from importlib_metadata import PathDistribution
from numpy import pad
import torch 
from torch import nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
import gensim
import os, json
from segword import BIMMSegment
from utils import convt
import copy


#jieba.initialize()
path = r'D:\ai-risk\aliwentian\word2vec\word2vec.bin'
path = '../word2vec/word2vec.bin'
word2vec = gensim.models.KeyedVectors.load(path)

#取腾讯词向量的前100万个
# for word in  word2vec.index_to_key[:1500000]:
#     if word not in jieba.dt.FREQ:
#         jieba.add_word(word)
        
#id2word = {i+2:j  for i, j in enumerate(word2vec.index_to_key[:1000000])}  # 0: <pad> 1:<unk>
#word2id = {j: i for i, j in id2word.items()}

# word2vec = word2vec.vectors[:1000000]
# word_dim = word2vec.shape[1]
# word2vec = np.concatenate([np.zeros((2, word_dim)), word2vec])

seg = BIMMSegment(word2vec.index_to_key[:1000000])

class MyDataSet():
    def __init__(self, path, word2id, tokenizer=None, batch_size=32, maxlen=64, negs_num=64, hard_num=3, hardneguse=False):
        self.queries = self.readQuery(path)
        self.docs = self.readDoc(path)
        self.labels = self.readLabel(path)
        self.devqueries = self.readQuery(path, mode='dev')
        self.keywords = self.readKeyWords(path)
        self.tokenizer = tokenizer
        self.step = 0
        self.batch_size = batch_size
        self.word2id = word2id
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
    
    def words2charid(self, words):
        ids = []
        charids = []
        for word in words:
            id_ = self.tokenizer.encode(word, add_special_tokens=False)
            charids.extend(id_)
            ids.extend([self.word2id.get(word, 1)] * len(id_))
        return ids, charids 
    
    def encode(self, s):
        s = s.lower()
        s = convt(s)
        id, idchar = [], []
        for sub in s.split():
            sub = seg.lcut(sub)
            sub, subchar = self.words2charid(sub)
            id.extend(sub)
            idchar.extend(subchar)
            
        id = [101] + id
        mask = [1] * len(id)
        idchar = [101] + idchar
        assert len(id) == len(idchar), 'the word is wrong'
        idchar = torch.tensor(idchar)
        id = torch.tensor(id)
        mask = torch.tensor(mask)
        return idchar, mask, id
    
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
            queries_bert, docs_bert, negs_bert = [], [], []
            queries_bertmask, docs_bertmask, negs_bertmask = [], [], []
            q_words, d_words, negs_words = [], [], []
            hardids = []
            for index in permutation[start:end]:
                query = self.queries[self.labels[index][0]]
                doc = self.docs[self.labels[index][1]]
               
                queries_ids, queries_mask, q_word = self.encode(query)
                doc_ids, doc_mask, d_word = self.encode(doc)
                queries_bert.append(queries_ids)
                queries_bertmask.append(queries_mask)
                docs_bert.append(doc_ids)
                docs_bertmask.append(doc_mask)
                q_words.append(q_word)
                d_words.append(d_word)
                
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
                
                negs_ids, negs_mask, negs_word = self.encode(negdoc)
                negs_bert.append(negs_ids)
                negs_bertmask.append(negs_mask)
                negs_words.append(negs_word)
                
            if self.hardneguse:
                hard_bert, hard_bertmask = [], []
                hard_words = []
                for index in range(len(hardids)):
                    for id in hardids[index]:
                        doc = self.docs[id]
                        doc_ids, doc_mask, doc_words = self.encode(doc)
                        hard_bert.append(doc_ids)
                        hard_bertmask.append(doc_mask)
                        hard_words.append(doc_words)
                        
                hard_bert = pad_sequence(hard_bert, batch_first=True)
                hard_bertmask = pad_sequence(hard_bertmask, batch_first=True)
                hard_bert = hard_bert.reshape(len(hardids), self.hard_num, -1)
                hard_bertmask = hard_bertmask.reshape(len(hardids), self.hard_num, -1)
                hard_words = pad_sequence(hard_words, batch_first=True)
                hard_words = hard_words.reshape(len(hardids), self.hard_num, -1)
            
          
            queries_bert = pad_sequence(queries_bert, batch_first=True)
            queries_bertmask = pad_sequence(queries_bertmask, batch_first=True)
            docs_bert = pad_sequence(docs_bert, batch_first=True)
            docs_bertmask = pad_sequence(docs_bertmask, batch_first=True)
            negs_bert = pad_sequence(negs_bert, batch_first=True)
            negs_bertmask = pad_sequence(negs_bertmask, batch_first=True)
            q_words = pad_sequence(q_words, batch_first=True)
            d_words = pad_sequence(d_words, batch_first=True)
            negs_words = pad_sequence(negs_words, batch_first=True)
            
            if self.hardneguse:
                yield queries_bert, docs_bert, negs_bert, queries_bertmask, docs_bertmask, negs_bertmask,  hard_bert, hard_bertmask, q_words, d_words, negs_words, hard_words
            else:
                yield queries_bert, docs_bert, negs_bert, queries_bertmask, docs_bertmask, negs_bertmask, q_words, d_words, negs_words
            if end == len(permutation):
                step = 0
                if mode == 'test':
                    break
            else:
                step += 1 
     
    def iter_queries(self, mode='train', shuffle=False):
        step = 0
        assert mode in ('train', 'test', 'dev'),   "mode is invalid"
        if mode == 'train':
            queries_origin = self.queries[1:]
            if shuffle:
                np.random.shuffle(queries_origin)
        if mode == 'test':
            queries_origin = []
            for index in self.test_permutation:
                queries_origin.append(self.queries[self.labels[index][0]])
        if mode == 'dev':
            queries_origin = self.devqueries
        while True:
            start = step * self.batch_size
            end = min(len(queries_origin), start + self.batch_size)
            queries_bert, queries_bertmask, q_words = [], [], []
            for index in range(start, end):
                query = queries_origin[index]
                bert_ids, bert_mask, bert_words = self.encode(query)
                queries_bert.append(bert_ids)
                queries_bertmask.append(bert_mask)
                q_words.append(bert_words)
                
            queries_bert = pad_sequence(queries_bert, batch_first=True)
            queries_bertmask = pad_sequence(queries_bertmask, batch_first=True)
            q_words = pad_sequence(q_words, batch_first=True)
            
            yield queries_bert, queries_bertmask, q_words
            if end == len(queries_origin):
                step = 0
                break
            else:
                step += 1 
     
    def iter_docs(self, shuffle=False):
        step = 0
        if shuffle:
            docs = copy.deepcopy(self.docs[1:])
            np.random.shuffle(docs)
        else:
            docs = self.docs[1:]
        while True:
            start = step * self.batch_size
            end = min(len(docs), start + self.batch_size)
            docs_bert, docs_bertmask, d_words = [], [], []
            for index in range(start, end): 
                doc = docs[index]
                doc_ids, doc_mask, d_word = self.encode(doc)
                docs_bert.append(doc_ids)
                docs_bertmask.append(doc_mask)
                d_words.append(d_word)
                
            docs_bert = pad_sequence(docs_bert, batch_first=True)
            docs_bertmask = pad_sequence(docs_bertmask, batch_first=True)
            d_words = pad_sequence(d_words, batch_first=True)
            yield docs_bert, docs_bertmask, d_words
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

    def readKeyWords(self, path):
        with open(path + 'corpus_keywords.tsv', encoding='utf-8') as f:
            keywords = [['']] + [json.loads(line.strip().split('\t')[1]) for line in f.readlines()]
        return keywords
    
    def readHardSamples(self, path, mode='hard'):
        hardSamples = {}
        if not os.path.exists(path + 'hard_samples.txt') or mode != 'hard':
            pass
        else:
            with open(path + 'hard_samples.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    queryid, hardids = line.strip().split('\t')
                    hardSamples[int(queryid)] = [int(id) for id in hardids.split(',')[300:400]]  
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
