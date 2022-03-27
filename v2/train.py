import torch 
from torch import nn 
import torch.nn.functional as F
from model import Model
from dataset import MyDataSet
import gensim
import numpy as np
import json
from transformers import BertTokenizer, BertModel, BertConfig, AutoTokenizer
import faiss

path = r'D:\ai-risk\aliwentian\word2vec\word2vec.bin'
path = '../word2vec/word2vec.bin'

gpuflag = torch.cuda.is_available()
np.random.seed(2022)

datapath = 'D:\\ai-risk\\aliwentian\\data\\'
datapath = '../data/'

# hfl/chinese-roberta-wwm-ext
#tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
tokenizer = AutoTokenizer.from_pretrained("cyclone/simcse-chinese-roberta-wwm-ext")
dataset = MyDataSet(datapath, tokenizer, batch_size=32, negs_num=64, harduse=True, hardnum=1)
model = Model(hardnum=1)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)   # 1.4版本要加上

train_data = dataset.iter_permutation()
lrscheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)  # lr * epoch**setp
best_hits5, best_hits1 = 0.0, 0.0
best_mrr = 0.0

model.load_state_dict(torch.load(f'../model_v2/model_best.pt')) 
if gpuflag:
    model = model.cuda()

# 仿照真实的评估
def evaluate(dataset, model):
    docEmb = []
    for input in dataset.iter_docs():
        input = [d.cuda() for d in input]
        emb = model.Tower(input).cpu().tolist()
        docEmb.extend(emb) 
    print("doc emb generate complete!")    
    testqueryEmb = []
    for input in dataset.iter_queries(mode='test'):
        input = [d.cuda() for d in input]
        emb = model.Tower(input).cpu().tolist()
        testqueryEmb.extend(emb)
    docEmb = np.array(docEmb).astype(np.float32)
    testqueryEmb = np.array(testqueryEmb).astype(np.float32)
    d = 128
    docindex = faiss.IndexFlatL2(d)  
    docindex.add(docEmb)             
    k = 300                        # we want to see 4 nearest neighbors
    D, I = docindex.search(testqueryEmb, k)     # xq  
    mrr = 0
    for i in range(len(dataset.test_permutation)):
        index = dataset.test_permutation[i]
        queryid, docid = dataset.labels[index]
        queryid, docid = int(queryid), int(docid)
        try:
            score = list(I[i]).index(docid-1)
            mrr += 1.0 / score
        except:
            pass
    mrr = mrr / len(dataset.test_permutation)
    print(f"test dataset mrr : {mrr}")
    # 生成困难样本学习
    trainqueryEmb = []
    for input in dataset.iter_queries():
        input = [d.cuda() for d in input]
        emb = model.Tower(input).cpu().tolist()
        trainqueryEmb.extend(emb)
    trainqueryEmb = np.array(trainqueryEmb).astype(np.float32)
    D, I = docindex.search(trainqueryEmb, 100)  
    with open("../data/hard_samples.txt", 'w+', encoding='utf-8') as f:
        for index in range(len(I)):
            hard_samples = [str(id + 1) for id in list(I[index])]
            f.write(str(index+1) + '\t' + ','.join(hard_samples) + '\n' )
    return mrr

test_data = dataset.iter_permutation(mode='test')
model.eval()
with torch.no_grad():
    # mrr = evaluate(dataset, model)
    # if mrr > best_mrr:
    #     best_mrr = mrr
    #     torch.save(model.state_dict(), f'../model_v2/model_best.pt')
    sumhits5, sumhits1 = 0, 0
    for input in test_data:
        if gpuflag:
            input = [d.cuda() for d in input]
        hits5, hits1 = model.scores(input)
        sumhits5 += hits5
        sumhits1 += hits1
    best_hits5 = sumhits5/len(dataset.test_permutation)
    best_hits1 = sumhits1/len(dataset.test_permutation)
    
print(f"cur hits5: {best_hits5}, cur hits1: {best_hits1}")
    
for epoch in range(3):
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
    #lrscheduler.step()
        # evaluate
        if step % 200 == 199:
            test_data = dataset.iter_permutation(mode='test')
            model.eval()
            with torch.no_grad():
                # mrr = evaluate(dataset, model)
                # if mrr > best_mrr:
                #     best_mrr = mrr
                #     torch.save(model.state_dict(), f'../model_v2/model_best.pt')
                sumhits5, sumhits1 = 0, 0
                for input in test_data:
                    if gpuflag:
                        input = [d.cuda() for d in input]
                    hits5, hits1 = model.scores(input)
                    sumhits5 += hits5
                    sumhits1 += hits1
                curhits5 = sumhits5/len(dataset.test_permutation)
                curhits1 = sumhits1/len(dataset.test_permutation)
                if curhits5 > best_hits5:
                    # torch.save(model.state_dict(), f'../model_v2/model_best.pt')
                    best_hits5 = curhits5
                if curhits1 > best_hits1:
                    torch.save(model.state_dict(), f'../model_v2/model_best.pt')
                    best_hits1 = curhits1
                print(f"cur_hits5: {curhits5} ,  max hits5: {best_hits5}" )
                print(f"cur_hits1: {curhits1} ,  max hits1: {best_hits1}")
                torch.save(model.state_dict(), f'../model_v2/model_2.pt')
    
 
# 生成embedding文件
def generateEmbeddingFile(dataset, model):
    devqueryEmb = []
    for input in dataset.iter_queries(mode='dev'):
        input = [d.cuda() for d in input]
        emb = model.Tower(input).cpu().tolist()
        devqueryEmb.extend(emb)
    
    with open('../submit/query_embedding', 'w+', encoding='utf-8') as f:
        for i in range(len(devqueryEmb)):
            emb = [str(round(number, 8)) for number in devqueryEmb[i]]
            f.write( str(i+200001) + '\t' + ','.join(emb) + '\n')
        
    docEmb = []
    for input in dataset.iter_docs():
        input = [d.cuda() for d in input]
        emb = model.Tower(input).cpu().tolist()
        docEmb.extend(emb)
    
    with open("../submit/doc_embedding", 'w+', encoding='utf-8') as f:
        for i in range(len(docEmb)):
            emb = [str(round(number, 8)) for number in docEmb[i]]
            f.write( str(i+1) + '\t' + ','.join(emb) + '\n' )
            
    trainqueryEmb = []
    for input in dataset.iter_queries():
        input = [d.cuda() for d in input]
        emb = model.Tower(input).cpu().tolist()
        trainqueryEmb.extend(emb)
    
    with open('../submit/trainquery_embedding', 'w+', encoding='utf-8') as f:
        for i in range(len(trainqueryEmb)):
            emb = [str(round(number, 8)) for number in trainqueryEmb[i]]
            f.write( str(i+1) + '\t' + ','.join(emb) + '\n')
            

model.load_state_dict(torch.load(f'../model_v2/model_best.pt'))  
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

    k = 800                         # we want to see 4 nearest neighbors
    D, I = docindex.search(trainqueryEmb, k)     # xq为query embedding, 大小为(10000,128)
    #print(I[:10])                   # neighbors of the 5 first queries

    with open('../data/qrels.train.tsv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip().split('\t') for line in lines]
        
    mrr = 0
    for queryid, docid in lines:
        queryid, docid = int(queryid), int(docid)
        try:
            score = list(I[queryid-1])[:10].index(docid-1)
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
    

