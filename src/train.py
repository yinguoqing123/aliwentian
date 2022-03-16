from os import device_encoding
from pydoc import doc
from random import betavariate
from turtle import forward
import torch 
from torch import nn 
import torch.nn.functional as F
from model import Model
from dataset import MyDataSet
import gensim
import numpy as np

path = r'D:\ai-risk\aliwentian\word2vec\word2vec.bin'
path = '../word2vec/word2vec.bin'
word2vec = gensim.models.KeyedVectors.load(path)

gpuflag = torch.cuda.is_available()
        
id2word = {i+2:j  for i, j in enumerate(word2vec.index_to_key[:1000000])}  # 0: <pad> 1:<unk>
word2id = {j: i for i, j in id2word.items()}

word2vec = word2vec.vectors[:1000000]
word_dim = word2vec.shape[1]
word2vec = np.concatenate([np.zeros((2, word_dim)), word2vec])

datapath = 'D:\\ai-risk\\aliwentian\\data\\'
datapath = '../data/'
dataset = MyDataSet(datapath, word2id)
model = Model(len(word2id), word2vec.shape[1], word2vec)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_data = dataset.iter_train_permutation()
lrscheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)  # lr * epoch**setp
best_hits10, best_hits1 = 0.0, 0.0

if gpuflag:
    model = model.cuda()

for epoch in range(20):
    running_loss = 0
    for step in range(len(dataset)):
        queries, docs, negs = next(train_data)
        model.train()
        if gpuflag:
            queries, docs, negs = queries.cuda(), docs.cuda(), negs.cuda()
        loss = model((queries, docs, negs))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if step % 100 == 99:
            print(f"Epoch {epoch+1}, step {step+1}: {running_loss}")
            running_loss = 0 
    lrscheduler.step()
    # evaluate
    test_data = dataset.iter_test_permutation()
    model.eval()
    with torch.no_grad():
        sumhits10, sumhits1 = 0, 0
        for input in test_data:
            if gpuflag:
                input = input.cuda()
            hits10, hits1 = model.scores(input)
            sumhits10 += hits10
            sumhits1 += hits1
            curhits10 = sumhits10/len(dataset.test_permutation)
            curhits1 = sumhits1/len(dataset.test_permutation)
            if curhits10 > best_hits10:
                torch.save(model.state_dict(), f'../model/model.pt')
                best_hits10 = curhits10
            if curhits1 > best_hits1:
                best_hits1 = curhits1
        print(f"cur_hits10: {curhits10} ,  max hits10: {best_hits10}" )
        print(f"cur_hits1: {curhits1} ,  max hits1: {best_hits1}")
        
        
# 生成embedding文件

def generateEmbeddingFile(dataset, model):
    devqueryEmb = []
    for queries in dataset.iter_dev_queries():
        emb = model.queryTower(queries).tolist()
        devqueryEmb.extend(emb)
    
    with open('../submit/query_embedding', 'w', encoding='utf-8') as f:
        for i in range(len(devqueryEmb)):
            emb = [str(round(number, 6)) for number in devqueryEmb[i]]
            f.wirte( str(i+200000) + '\t' + ','.join(emb) + '\n')
        
    docEmb = []
    for docs in dataset.iter_docs():
        emb = model.docTower(docs).tolist()
        docEmb.extend(emb)
    
    with open("../submit/doc_embedding", 'w', encoding='utf-8') as f:
        for i in range(len(docEmb)):
            emb = [str(round(number, 6)) for number in docEmb[i]]
            f.write( str(i+1) + '\t' + ','.join(emb) + '\n' )

model.load_state_dict(torch.load(f'../model/model.pt'))  
generateEmbeddingFile(dataset)
                
        






