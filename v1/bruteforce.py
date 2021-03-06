# 暴力评估
import faiss
import numpy as np

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
        
devqueryEmb = []
with open("../submit/query_embedding", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split('\t')[1].split(',')
        line = [float(n) for n in line]
        devqueryEmb.append(line)

trainqueryEmb = np.array(trainqueryEmb).astype(np.float32)
docEmb = np.array(docEmb).astype(np.float32)
devqueryEmb = np.array(devqueryEmb).astype(np.float32)

d = 128
docindex = faiss.IndexFlatL2(d)   # build the index, d = 128, 为dimension
docindex.add(docEmb)                  # add vectors to the index, xb 为 (100000,128)大小的numpy
print(docindex.ntotal)            # 索引中向量的数量, 输出100000

k = 10                         # we want to see 4 nearest neighbors
D, I = docindex.search(trainqueryEmb, k)     # xq为query embedding, 大小为(10000,128)
#print(I[:10])                   # neighbors of the 5 first queries

with open('../data/qrels.train.tsv', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines = [line.strip().split('\t') for line in lines]

   
lines = lines[:5000]
mrr = 0
for queryid, docid in lines:
    queryid, docid = int(queryid), int(docid)
    try:
        score = list(I[queryid-1]).index(docid-1)
        mrr += 1.0 / score
    except:
        pass

mrr = mrr / 5000
print(f"mrr : {mrr}")
        
D, I = docindex.search(devqueryEmb, k)