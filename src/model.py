import torch 
from torch import nn 
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, nums_word, nums_char, dims, embedding_init=None) -> None:
        super().__init__()
        self.wordemb = nn.Embedding(nums_word, 100)
        self.charemb = nn.Embedding(nums_char, dims)
        self.word2char = nn.Linear(100, dims)
        self.queryatt = AttentionPooling1D(dims)
        self.docatt = AttentionPooling1D(dims)
        #self.qd1 = nn.Linear(dims, 128)
        #self.dd1 = nn.Linear(dims, 128)
        self.cretirion = nn.CrossEntropyLoss()
        if embedding_init is not None:
            self.wordemb.weight.data = torch.tensor(embedding_init, dtype=torch.float)
            self.wordemb.weight.requires_grad = False
        
    def queryTower(self, input):
        """
        param: input  
        """
        query, querychar = input
        mask = torch.gt(querychar, 0).unsqueeze(dim=-1).float()
        querywordEmb = self.wordemb(query)
        querywordEmb = self.word2char(querywordEmb)
        querycharEmb = self.charemb(querychar)
        #queryEmb = torch.sum(queryEmb * mask, dim=-2)
        #queryEmb = queryEmb / torch.sum(mask, dim=-2)
        queryEmb = querywordEmb + querycharEmb
        queryEmb = self.queryatt(queryEmb, mask)
        #queryEmb = self.qd1(queryEmb)
        queryEmb = F.normalize(queryEmb, dim=-1)
        #queryEmb = torch.relu(self.qd1(queryEmb))
        return queryEmb
        
    def docTower(self, input):
        doc, docchar = input
        mask = torch.gt(doc, 0).unsqueeze(dim=-1).float()
        docwordEmb = self.wordemb(doc)
        docwordEmb = self.word2char(docwordEmb)
        doccharemb = self.charemb(docchar)
        #docEmb = torch.sum(docEmb * mask, dim=-2)
        #docEmb = docEmb / torch.sum(mask, dim=-2)
        docEmb = docwordEmb + doccharemb
        docEmb = self.docatt(docEmb, mask)
        #docEmb = self.dd1(docEmb)
        docEmb = F.normalize(docEmb, dim=-1)
        #docEmb = torch.relu(self.dd1(docEmb))
        return docEmb
        
    def forward(self, input):
        query, querychar, doc, docchar, neg, negchar = input
        maxlen = max(doc.shape[1], neg.shape[1])
        doc = F.pad(doc, (0, maxlen-doc.shape[1]))
        docchar = F.pad(docchar, (0, maxlen-doc.shape[1]))
        neg = F.pad(neg, (0, maxlen-neg.shape[1]))
        negchar = F.pad(negchar, (0, maxlen-neg.shape[1]))
        queryEmb = self.queryTower((query, querychar))
        docEmb = self.docTower((doc, docchar))
        negEmb = self.docTower((neg, negchar))
        docAndNegEmb = torch.cat([docEmb, negEmb], dim=0)
        loss = self.loss(queryEmb, docAndNegEmb)
        return loss
    
    def loss(self, queryEmb, docEmb):
        scores = torch.matmul(queryEmb, docEmb.t()) * 5  # (batch_size, batch_size+neg_nums)
        labels = torch.arange(scores.shape[0]).cuda()
        loss = self.cretirion(scores, labels)
        return loss
    
    def scores(self, input):
        query, querychar, doc, docchar, neg, negchar = input
        queryEmb = self.queryTower((query, querychar))
        maxlen = max(doc.shape[1], neg.shape[1])
        doc = F.pad(doc, (0, maxlen-doc.shape[1]))
        docchar = F.pad(docchar, (0, maxlen-doc.shape[1]))
        neg = F.pad(neg, (0, maxlen-neg.shape[1]))
        negchar = F.pad(negchar, (0, maxlen-neg.shape[1]))
        docEmb = self.docTower((doc, docchar))
        negEmb = self.docTower((neg, negchar))
        docAndNegEmb = torch.cat([docEmb, negEmb], dim=0)
        scores = torch.matmul(queryEmb, docAndNegEmb.t())    # (batch_size, batch_size+negnums)
        _, indices = scores.topk(10)   # (b, 10)
        label = torch.arange(queryEmb.shape[0]).view(-1, 1).expand_as(indices).cuda()
        hits = (label == indices).nonzero() 
        if  len(hits) == 0:
            return 0
        else:
            hits = (hits[:, 1] + 1).float()
            hits = torch.reciprocal(hits)
            hits10 = torch.sum(hits)
        indices = scores.argmax(dim=-1)
        hits1 = torch.sum(torch.eq(indices, torch.arange(queryEmb.shape[0]).cuda()))
        return hits10.item(), hits1.item()
         
        
class AttentionPooling1D(nn.Module):
    """通过加性Attention,将向量序列融合为一个定长向量
    """
    def __init__(self, in_features,  **kwargs):
        super(AttentionPooling1D, self).__init__(**kwargs)
        self.in_features = in_features # 词向量维度
        self.k_dense = nn.Linear(self.in_features, self.in_features, bias=False)
        self.o_dense = nn.Linear(self.in_features, 1, bias=False)
    def forward(self, xo, mask):
        x = self.k_dense(xo)
        x = self.o_dense(torch.tanh(x))  # N, s, 1
        x = x - (1 - mask) * 1e12
        x = F.softmax(x, dim=-2)  # N, w, 1
        return torch.sum(x * xo, dim=-2) # N*emd