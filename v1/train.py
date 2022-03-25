import torch 
from torch import nn 
import torch.nn.functional as F
from model import Model
from dataset import MyDataSet
import gensim
import numpy as np
import json
from transformers import BertTokenizer
import faiss
import random

np.random.seed(2022)
path = r'D:\ai-risk\aliwentian\word2vec\word2vec.bin'
path = '../word2vec/word2vec.bin'
word2vec = gensim.models.KeyedVectors.load(path)

gpuflag = torch.cuda.is_available()
        
id2word = {i+2:j  for i, j in enumerate(word2vec.index_to_key[:1000000])}  # 0: <pad> 1:<unk>
id2word[0] = '<pad>'
id2word[1] = '<unk>'
word2id = {j: i for i, j in id2word.items()}

char2id, id2char = json.load(open("charidmap.json"))

word2vec = word2vec.vectors[:1000000]
word_dim = word2vec.shape[1]
word2vec = np.concatenate([np.zeros((2, word_dim)), word2vec])

datapath = 'D:\\ai-risk\\aliwentian\\data\\'
datapath = '../data/'
tokenizer = BertTokenizer.from_pretrained("hfl/rbt3")
dataset = MyDataSet(datapath, word2id, char2id, tokenizer=tokenizer, batch_size=128, hard_num=5)
model = Model(len(word2id), len(char2id), 128, embedding_init = word2vec)

bert_parameters = list(model.bert.parameters())
other_parameters = []
for name, param in model.named_parameters():
    if 'bert' not in name and param.requires_grad:
        other_parameters.append(param)


p = [{'params': bert_parameters, 'lr': 3e-5}, {'params': other_parameters, 'lr': 1e-3}]
optimizer = torch.optim.Adam(p)   # 1.4版本要加上

train_data = dataset.iter_permutation()
lrscheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.8)  # lr * epoch**setp
best_hits10, best_hits1 = 0.0, 0.0
best_mrr = 0.0

#model.load_state_dict(torch.load(f'../model/model_best.pt')) 
if gpuflag:
    model = model.cuda()

# 仿照真实的评估
def evaluate(dataset, model):
    docEmb = []
    for input in dataset.iter_docs():
        input = [d.cuda() for d in input]
        emb = model.docTower(input).cpu().tolist()
        docEmb.extend(emb)     
    testqueryEmb = []
    for input in dataset.iter_queries(mode='test'):
        input = [d.cuda() for d in input]
        emb = model.queryTower(input).cpu().tolist()
        testqueryEmb.extend(emb)
    docEmb = np.array(docEmb).astype(np.float32)
    testqueryEmb = np.array(testqueryEmb).astype(np.float32)
    d = 128
    docindex = faiss.IndexFlatL2(d)  
    docindex.add(docEmb)             
    k = 100                       # we want to see 4 nearest neighbors
    D, I = docindex.search(testqueryEmb, k)     # xq  
    mrr = 0
    f = open('../data/test_query_doc.txt', 'w+', encoding='utf-8')
    for i in range(len(dataset.test_permutation)):
        index = dataset.test_permutation[i]
        queryid, docid = dataset.labels[index]
        queryid, docid = int(queryid), int(docid)
        matchdoc = [str(id + 1) for id in list(I[i])]
        f.write(str(queryid) + '\t' + ','.join(matchdoc) + '\n' ) 
        try:
            score = list(I[i]).index(docid-1)
            mrr += 1.0 / score
        except:
            pass
    mrr = mrr / len(dataset.test_permutation)
    print(f"test dataset mrr : {mrr}")
    f.close()
    # 生成困难样本学习
    trainqueryEmb = []
    for input in dataset.iter_queries():
        input = [d.cuda() for d in input]
        emb = model.queryTower(input).cpu().tolist()
        trainqueryEmb.extend(emb)
    trainqueryEmb = np.array(trainqueryEmb).astype(np.float32)
    D, I = docindex.search(trainqueryEmb, 200)  
    with open("../data/hard_samples.txt", 'w+', encoding='utf-8') as f:
        for index in range(len(I)):
            hard_samples = [str(id + 1) for id in list(I[index])]
            f.write(str(index+1) + '\t' + ','.join(hard_samples) + '\n' ) 
    return mrr

#dataset.readHardSamples(path, mode='random')
for epoch in range(20):
    running_loss = 0
    for step in range(len(dataset)):
        input = next(train_data)
        model.train()
        if gpuflag:
            input = [f.cuda() for f in input]
        loss = model(input)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if step % 100 == 99:
            print(f"Epoch {epoch+1}, step {step+1}: {running_loss}")
            running_loss = 0 
    lrscheduler.step()
    # evaluate
    test_data = dataset.iter_permutation(mode='test')
    model.eval()
    with torch.no_grad():
        # if epoch > 3:
        #     mrr = evaluate(dataset, model)
        #     if mrr > best_mrr:
        #         best_mrr = mrr
        #         print(f"best mrr: {best_mrr}")
            # torch.save(model.state_dict(), f'../model/model_best.pt')
        sumhits10, sumhits1 = 0, 0
        for input in test_data:
            if gpuflag:
                input = [d.cuda() for d in input]
            hits10, hits1 = model.scores(input)
            sumhits10 += hits10
            sumhits1 += hits1
        curhits10 = sumhits10/len(dataset.test_permutation)
        curhits1 = sumhits1/len(dataset.test_permutation)
        if curhits10 > best_hits10:
            # torch.save(model.state_dict(), f'../model/model_best.pt')
            best_hits10 = curhits10
        if curhits1 > best_hits1:
            best_hits1 = curhits1
            torch.save(model.state_dict(), f'../model/model_best.pt')
        print(f"cur_hits10: {curhits10} ,  max hits10: {best_hits10}" )
        print(f"cur_hits1: {curhits1} ,  max hits1: {best_hits1}")
        torch.save(model.state_dict(), f'../model/model.pt')
        # 更新hard_sample
        # if epoch < 3:
        #     dataset.readHardSamples(path, mode='random')
        # else:
        #     dataset.readHardSamples(path, mode='hard')
        # if epoch > 5:
        #     dataset.hardneguse = True
        # dataset.readHardSamples(datapath, mode='hard') 
    

 
# 生成embedding文件
def generateEmbeddingFile(dataset, model):
    devqueryEmb = []
    for input in dataset.iter_queries(mode='dev'):
        input = [d.cuda() for d in input]
        emb = model.queryTower(input).cpu().tolist()
        devqueryEmb.extend(emb)
    
    with open('../submit/query_embedding', 'w+', encoding='utf-8') as f:
        for i in range(len(devqueryEmb)):
            emb = [str(round(number, 8)) for number in devqueryEmb[i]]
            f.write( str(i+200001) + '\t' + ','.join(emb) + '\n')
        
    docEmb = []
    for input in dataset.iter_docs():
        input = [d.cuda() for d in input]
        emb = model.docTower(input).cpu().tolist()
        docEmb.extend(emb)
    
    with open("../submit/doc_embedding", 'w+', encoding='utf-8') as f:
        for i in range(len(docEmb)):
            emb = [str(round(number, 8)) for number in docEmb[i]]
            f.write( str(i+1) + '\t' + ','.join(emb) + '\n' )
            
    trainqueryEmb = []
    for input in dataset.iter_queries():
        input = [d.cuda() for d in input]
        emb = model.queryTower(input).cpu().tolist()
        trainqueryEmb.extend(emb)
    
    with open('../submit/trainquery_embedding', 'w+', encoding='utf-8') as f:
        for i in range(len(trainqueryEmb)):
            emb = [str(round(number, 8)) for number in trainqueryEmb[i]]
            f.write( str(i+1) + '\t' + ','.join(emb) + '\n')

model.load_state_dict(torch.load(f'../model/model_best.pt'))  
generateEmbeddingFile(dataset, model)


def generateHardSample():
    trainqueryEmb = []
    with open("../submit/trainquery_embedding", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')[1].split(',')
            line = [float(n) for n in line]
            trainqueryEmb.append(line)

    docEmb = []
    with open("../submit/doc_embedding", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')[1].split(',')
            line = [float(n) for n in line]
            docEmb.append(line)
            
    trainqueryEmb = np.array(trainqueryEmb).astype(np.float32)
    docEmb = np.array(docEmb).astype(np.float32)

    d = 128
    docindex = faiss.IndexFlatL2(d)   # build the index, d = 128, 为dimension
    docindex.add(docEmb)                  # add vectors to the index, xb 为 (100000,128)大小的numpy
    print(docindex.ntotal)            # 索引中向量的数量, 输出100000

    k = 600                         # we want to see 4 nearest neighbors
    D, I = docindex.search(trainqueryEmb, k)     # xq为query embedding, 大小为(10000,128)
    #print(I[:10])                   # neighbors of the 5 first queries

    with open('../data/qrels.train.tsv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip().split('\t') for line in lines]
        
    mrr = 0
    for queryid, docid in lines:
        queryid, docid = int(queryid), int(docid)
        try:
            score = list(I[queryid-1]).index(docid-1)
            mrr += 1.0 / score
        except:
            pass

    mrr = mrr / 100000
    print(f"all data mrr : {mrr}")

    with open("../data/hard_samples.txt", 'w+', encoding='utf-8') as f:
        for index in range(len(I)):
            hard_samples = [str(id + 1) for id in list(I[index])]
            f.write(str(index+1) + '\t' + ','.join(hard_samples) + '\n' )
                
generateHardSample()