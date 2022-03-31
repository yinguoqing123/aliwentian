import torch 
from torch import nn 
import torch.nn.functional as F
from model import Model
from dataset import MyDataSet
import gensim
import numpy as np
import json
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
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
harduse = False
#tokenizer = BertTokenizer.from_pretrained("hfl/rbt3")
# huawei-noah/TinyBERT_6L_zh
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
dataset = MyDataSet(datapath, word2id, char2id, tokenizer=tokenizer, batch_size=40, negs_num=32, hard_num=1, hardneguse=harduse)
model = Model(len(word2id), len(char2id), 128, harduse=harduse, embedding_init = word2vec)

# state_dict = torch.load(f'../model/model_best.pt')
# model.load_state_dict(state_dict, strict=False) 

bert_parameters = list(model.bert.parameters())
other_parameters = []
for name, param in model.named_parameters():
    if 'bert' not in name and param.requires_grad:
        other_parameters.append(param)


p = [{'params': bert_parameters, 'lr': 2e-5}, {'params': other_parameters, 'lr': 3e-4}]
optimizer = torch.optim.Adam(p)   # 1.4版本要加上

print(other_parameters)

# parameters = list(model.parameters())
# optimizer = torch.optim.Adam(parameters, lr=1e-3)

train_data = dataset.iter_permutation()
lrscheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)  # lr * epoch**setp
best_hits5, best_hits1 = 0.0, 0.0
best_mrr = 0.0

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

#dataset.readHardSamples(path, mode='random')
for epoch in range(3):
    running_loss = 0
    for step in range(len(dataset)):
        input = next(train_data)
        model.train()
        if gpuflag:
            input = [f.cuda() for f in input]
        loss, hardids = model(input)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        torch.cuda.empty_cache()
        # hardids 再训练一次
        if epoch >= 4:
            hardneg, hardcharneg = [], []
            hardnegs_bert, hardnegs_bertmask = [], []
            hardneg_words = []
            doc, docchar, neg, negchar = input[2:6]
            docs_bert, docs_bertmask = input[7], input[10]
            negs_bert, negs_bertmask = input[8], input[11]
            q_words, d_words, negs_words = input[-3:]
            b = doc.shape[0]
            for index in hardids.flatten().cpu().tolist():
                if index < b:
                    hardneg.append(doc[index])
                    hardcharneg.append(docchar[index])
                    hardnegs_bert.append(docs_bert[index])
                    hardnegs_bertmask.append(docs_bertmask[index])
                    hardneg_words.append(d_words[index])
                else:
                    hardneg.append(neg[index-b])
                    hardcharneg.append(negchar[index-b])
                    hardnegs_bert.append(negs_bert[index-b])
                    hardnegs_bertmask.append(negs_bertmask[index-b])
                    hardneg_words.append(negs_words[index-b])
            hardneg = pad_sequence(hardneg, batch_first=True)
            hardcharneg = pad_sequence(hardcharneg, batch_first=True)
            hardnegs_bert = pad_sequence(hardnegs_bert, batch_first=True)
            hardnegs_bertmask = pad_sequence(hardnegs_bertmask, batch_first=True)
            hardneg_words = pad_sequence(hardneg_words, batch_first=True)
            input = input[:4] + [hardneg, hardcharneg] + input[6:8] +  [hardnegs_bert] + input[9:11] + [hardnegs_bertmask] + [q_words,  d_words, hardneg_words]
            loss = model(input, mode='hard')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
                   
        torch.cuda.empty_cache()

        if step % 100 == 99:
            print(f"Epoch {epoch+1}, step {step+1}: {running_loss}")
            running_loss = 0 

        if step % 300 == 299:
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
                    # torch.save(model.state_dict(), f'../model/model_best.pt')
                    best_hits5 = curhits5
                if curhits1 > best_hits1:
                    best_hits1 = curhits1
                    torch.save(model.state_dict(), f'../model/model_best.pt')
                print(f"cur_hits5: {curhits5} ,  max hits5: {best_hits5}" )
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
    lrscheduler.step()
    

 
# 生成embedding文件
def generateEmbeddingFile(dataset, model):
    model.eval()
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
            score = list(I[queryid-1][:10]).index(docid-1)
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