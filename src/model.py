import torch 
from torch import nn 
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, nums, dims, embedding_init=None) -> None:
        super().__init__()
        self.emb = nn.Embedding(nums, dims)
        self.qd1 = nn.Linear(dims, 128)
        self.dd1 = nn.Linear(dims, 128)
        self.cretirion = nn.CrossEntropyLoss()
        if embedding_init is not None:
            self.emb.weight.data = torch.tensor(embedding_init, dtype=torch.float)
        
    def queryTower(self, input):
        """
        param: input  
        """
        mask = torch.greater(input, 0).unsqueeze(dim=-1)
        queryEmb = self.emb(input)
        queryEmb = torch.sum(queryEmb * mask, dim=-2)
        queryEmb = queryEmb / torch.sum(mask, dim=-2)
        queryEmb = torch.relu(self.qd1(queryEmb))
        return queryEmb
        
    def docTower(self, input):
        mask = torch.greater(input, 0).unsqueeze(dim=-1)
        docEmb = self.emb(input)
        docEmb = torch.sum(docEmb * mask, dim=-2)
        docEmb = docEmb / torch.sum(mask, dim=-2)
        docEmb = torch.relu(self.dd1(docEmb))
        return docEmb
        
    def forward(self, input):
        query, doc, negs = input
        maxlen = max(doc.shape[1], negs.shape[1])
        doc = F.pad(doc, (0, 0, 0, maxlen-doc.shape[1]))
        negs = F.pad(negs, (0, 0, 0, maxlen-negs.shape[1]))
        queryEmb = self.queryTower(query)
        docAndNegEmb = self.docTower(torch.concat([doc, negs], dim=0))
        loss = self.loss(queryEmb, docAndNegEmb)
        return loss
    
    def loss(self, queryEmb, docEmb):
        queryEmb = F.normalize(queryEmb, dim=-1)
        docEmb = F.normalize(docEmb, dim=-1)
        scores = torch.matmul(queryEmb, docEmb.t())   # (batch_size, batch_size+neg_nums)
        labels = torch.arange(scores.shape[0])
        loss = self.cretirion(scores, labels)
        return loss

    def scores(self, input):
        query, doc, negs = input
        queryEmb = self.queryTower(query)
        maxlen = max(doc.shape[1], negs.shape[1])
        doc = F.pad(doc, (0, maxlen-doc.shape[1]))
        negs = F.pad(negs, (0, maxlen-negs.shape[1]))
        docAndNegEmb = self.docTower(torch.concat([doc, negs], dim=0))
        queryEmb = F.normalize(queryEmb, dim=-1)
        docAndNegEmb = F.normalize(docAndNegEmb, dim=-1)
        scores = torch.matmul(queryEmb, docAndNegEmb.t())   # (batch_size, batch_size)
        scores = scores.argmax(dim=-1)
        tp = torch.sum(torch.eq(scores, torch.arange(queryEmb.shape[0])))
        accuracy = tp / queryEmb.shape[0]
        return accuracy.item()
        
        
        
class AttentionPooling1D(nn.Module):
    """通过加性Attention,将向量序列融合为一个定长向量
    """
    def __init__(self, in_features,  **kwargs):
        super(AttentionPooling1D, self).__init__(**kwargs)
        self.in_features = in_features # 词向量维度
        self.k_dense = nn.Linear(self.in_features, self.in_features, bias=False)
        self.o_dense = nn.Linear(self.in_features, 1, bias=False)
    def forward(self, inputs):
        xo, mask = inputs
        mask = mask.unsqueeze(-1)
        x = self.k_dense(xo)
        x = self.o_dense(torch.tanh(x))  # N, s, 1
        x = x - (1 - mask) * 1e12
        x = F.softmax(x, dim=-2)  # N, w, 1
        return torch.sum(x * xo, dim=-2) # N*emd