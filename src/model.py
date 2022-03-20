import torch 
from torch import nn 
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, nums_word, nums_char, dims, hard_num, embedding_init=None) -> None:
        super().__init__()
        self.wordemb = nn.Embedding(nums_word, 100)
        self.charemb = nn.Embedding(nums_char, dims)
        self.word2char = nn.Linear(100, dims)
        self.queryatt = AttentionPooling1D(dims)
        self.docatt = AttentionPooling1D(dims)
        self.conv1d = nn.Conv1d(dims, dims, kernel_size=3, dilation=1, padding=1)
        self.conv1d2 = nn.Conv1d(dims, dims, kernel_size=3, dilation=1, padding=1)
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
        queryEmb = queryEmb.permute(0, 2, 1)
        queryEmb = self.conv1d(queryEmb)
        queryEmb = self.conv1d2(queryEmb)
        queryEmb = queryEmb.permute(0, 2, 1)
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
        docEmb = docEmb.permute(0, 2, 1)
        docEmb= torch.self.conv1d(docEmb)
        docEmb = torch.self.conv1d2(docEmb)
        docEmb = docEmb.permute(0, 2, 1)
        docEmb = self.docatt(docEmb, mask)
        #docEmb = self.dd1(docEmb)
        docEmb = F.normalize(docEmb, dim=-1)
        #docEmb = torch.relu(self.dd1(docEmb))
        return docEmb
        
    def forward(self, input):
        query, querychar, doc, docchar, neg, negchar, hardneg, hardneg_char = input
        # maxlen = max(doc.shape[1], neg.shape[1])
        # doc = F.pad(doc, (0, maxlen-doc.shape[1]))
        # docchar = F.pad(docchar, (0, maxlen-docchar.shape[1]))
        # neg = F.pad(neg, (0, maxlen-neg.shape[1]))
        # negchar = F.pad(negchar, (0, maxlen-negchar.shape[1]))
        queryEmb = self.queryTower((query, querychar))
        docEmb = self.docTower((doc, docchar))
        negEmb = self.docTower((neg, negchar))
        # reshape
        b, hard_num, s = hardneg.shape
        hardneg = hardneg.reshape(-1, s)
        hardneg_char = hardneg_char.reshape(-1, s)
        hardnegEmb = self.docTower((hardneg, hardneg_char))
        hardnegEmb = hardnegEmb.reshape(b, hard_num, -1)
        loss = self.loss(queryEmb, docEmb, negEmb, hardnegEmb)
        return loss
    
    def loss(self, queryEmb, docEmb, negEmb, hardnegEmb):
        docAndNegEmb = torch.cat([docEmb, negEmb], dim=0)
        scores = torch.matmul(queryEmb, docAndNegEmb.t()) * 20  # (batch_size, batch_size+neg_nums)
        labels = torch.arange(scores.shape[0]).cuda()
        loss1 = self.cretirion(scores, labels)
        scores = scores[:, :scores.shape[0]].t()
        loss2 = self.cretirion(scores, labels)
        loss3 = self.hardloss(queryEmb, docEmb, hardnegEmb)
        return loss1 + loss2 * 0.4 + 1.5*loss3
    
    def hardloss(self, queryEmb, docEmb, hardnegEmb):
        b = queryEmb.shape[0]
        docAndHardEmb = torch.cat([docEmb.unsqueeze(dim=1), hardnegEmb], dim=1) # b, 1+neg_num, dim 
        scores = queryEmb.unsqueeze(dim=1) * docAndHardEmb   # b, 1+neg_num, dim
        scores = torch.sum(scores, dim=-1)  # b, 1+neg_num
        pos_scores = scores[:, 0]
        neg_scores = scores[:, 1:]
        pos_scores_aux = pos_scores - 0.7  # 希望pos_scores的分数大于0.7
        pos_scores_aux = pos_scores_aux.unsqueeze(dim=1)
        pos_scores = pos_scores.view(-1, 1).expand_as(neg_scores)  # b, neg_num
        loss = torch.cat([torch.zeros(b, 1).cuda(), neg_scores - pos_scores, pos_scores_aux], dim=-1) * 20  # b, 1+neg_num
        loss = torch.logsumexp(neg_scores-pos_scores, dim=-1)  # b
        return loss.mean()
          
    def scores(self, input):
        query, querychar, doc, docchar, neg, negchar, hardneg, hardnegchar = input
        queryEmb = self.queryTower((query, querychar))
        # maxlen = max(doc.shape[1], neg.shape[1])
        # doc = F.pad(doc, (0, maxlen-doc.shape[1]))
        # docchar = F.pad(docchar, (0, maxlen-docchar.shape[1]))
        # neg = F.pad(neg, (0, maxlen-neg.shape[1]))
        # negchar = F.pad(negchar, (0, maxlen-negchar.shape[1]))
        docEmb = self.docTower((doc, docchar))
        negEmb = self.docTower((neg, negchar))
        b, hard_num, s = hardneg.shape
        hardneg = hardneg.reshape(-1, s)
        hardnegchar = hardnegchar.reshape(-1, s)
        hardnegEmb = self.docTower((hardneg, hardnegchar))
        hardnegEmb = hardnegEmb.reshape(b, hard_num, -1)
        docAndNegEmb = torch.cat([docEmb, negEmb], dim=0)
        scores = torch.matmul(queryEmb, docAndNegEmb.t())    # (batch_size, batch_size+negnums)
        scoreshard = torch.sum(queryEmb.unsqueeze(dim=1) * hardnegEmb, dim=-1)  # batch_size, hardnegnums
        scores = torch.cat([scores, scoreshard], dim=-1)  # batch_size, batch_size+negnum+hardnegnum
        _, indices = scores.topk(10)   # (b, 10)
        label = torch.arange(queryEmb.shape[0]).view(-1, 1).expand_as(indices).cuda()
        hits = (label == indices).nonzero() 
        if  len(hits) == 0:
            return 0, 0
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
    
class DilatedGatedConv1D(nn.Module):
    """膨胀门卷积(DGCNN)
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 dilation=1,
                 skip_connect=True,
                 drop_gate=None,
                 **kwargs):
        super(DilatedGatedConv1D, self).__init__(**kwargs)
        self.in_channels = in_channels  #词向量的维度
        self.out_channels = out_channels # 卷积后词向量的维度
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.skip_connect = skip_connect
        self.drop_gate = drop_gate
        self.conv1d = nn.Conv1d(
            self.in_channels,
            self.out_channels*2,
            self.kernel_size,
            dilation=self.dilation,
            padding='same'
        )
        if self.skip_connect and self.in_channels != self.out_channels:
            self.conv1d_1x1 = nn.Conv1d(self.in_channels, self.out_channels, padding='same')
        if self.drop_gate:
            self.dropout = nn.Dropout(drop_gate)
    def forward(self, input):
        xo = input
        mask = torch.gt(xo, 0).unsqueeze(dim=-1).float()
        x = xo * mask 
        x = self.conv1d(x)  
        x, g = x[:, :self.out_channels, ...], x[:, self.out_channels:, ...]
        if self.drop_gate:
            g = self.dropout(g)
        g = F.sigmoid(g)
        if self.skip_connect:
            if self.in_channels != self.out_channels:
                xo = self.conv1d_1x1(xo)
            return xo * (1 - g) + x * g
        else:
            return x * g * mask