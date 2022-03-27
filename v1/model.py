import torch 
from torch import nn 
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig
import math

class Model(nn.Module):
    def __init__(self, nums_word, nums_char, dims, harduse=False, embedding_init=None) -> None:
        super().__init__()
        self.config = BertConfig.from_pretrained("hfl/rbt3")  # (attention_probs_dropout_prob, hidden_dropout_prob)
        self.config.update({'output_hidden_states': True})
        self.bert = BertModel.from_pretrained("hfl/rbt3", config=self.config)
        self.tokenizer = BertTokenizer.from_pretrained("hfl/rbt3")
        self.wordemb = nn.Embedding(nums_word, 100)
        self.charemb = nn.Embedding(nums_char, dims)
        self.word2char = nn.Linear(100, dims)
        self.block = Block(dims)
        self.fusion = nn.Linear(dims*4, 4)
        self.mlp = nn.Linear(768, dims)
        # self.block = AttentionPooling1D(dims)
        self.position_emb = nn.Embedding(64, dims)
        self.positionalencoding1d(dims, 64)
        self.register_buffer("position_ids", torch.arange(64).expand((1, -1)))
        self.harduse = harduse
        #self.queryatt = AttentionPooling1D(dims)
        #self.docatt = AttentionPooling1D(dims)
        #self.conv1d = nn.Conv1d(dims, dims, kernel_size=3, dilation=1, padding=1)
        #self.conv1d2 = nn.Conv1d(dims, dims, kernel_size=3, dilation=1, padding=1)
        #self.qd1 = nn.Linear(dims, 128)
        #self.dd1 = nn.Linear(dims, 128)
        self.cretirion = nn.CrossEntropyLoss()
        if embedding_init is not None:
            self.wordemb.weight.data = torch.tensor(embedding_init, dtype=torch.float)
            self.wordemb.weight.requires_grad = False
            
    def embFusion(self, queryEmb1, queryEmb2, queryEmbbert1, queryEmbbert2):
        b = queryEmb1.shape[0]
        queryEmbbert = self.mlp(torch.cat([queryEmbbert1, queryEmbbert2], dim=0))
        queryEmbbert1, queryEmbbert2 = queryEmbbert[:b, :], queryEmbbert[b:, :]
        weight = self.fusion(torch.cat([queryEmb1, queryEmb2, queryEmbbert1, queryEmbbert2], dim=-1))
        weight = torch.softmax(weight, dim=-1) # (batch_size, 4)
        queryEmb = weight.unsqueeze(-1) * torch.stack([queryEmb1, queryEmb2, queryEmbbert1, queryEmbbert2], dim=1)
        queryEmb = torch.sum(queryEmb, dim=1)
        # queryEmb = queryEmb + queryEmbbert
        return queryEmb
        
    def queryTower(self, input):
        """
        param: input  
        """
        query, querychar, query_bert, query_bertmask = input
        mask = torch.gt(querychar, 0).unsqueeze(dim=-1).float()
        querywordEmb = self.wordemb(query)
        querywordEmb = self.word2char(querywordEmb)
        querycharEmb = self.charemb(querychar)
        positionEmb = self.position_emb(self.position_ids[:, :query.shape[1]])
        #queryEmb = torch.sum(queryEmb * mask, dim=-2)
        #queryEmb = queryEmb / torch.sum(mask, dim=-2)
        queryEmb = querywordEmb + querycharEmb 
        # queryEmb = queryEmb.permute(0, 2, 1)
        # queryEmb = self.conv1d(queryEmb)
        # queryEmb = self.conv1d2(queryEmb)
        # queryEmb = queryEmb.permute(0, 2, 1)
        # queryEmb = self.queryatt(queryEmb, mask)
        # queryEmb = self.qd1(queryEmb)
        queryEmb1 = self.block((queryEmb, mask)) 
        queryEmb2 = torch.sum(queryEmb * mask, dim=1) / torch.sum(mask.squeeze(), dim=1, keepdim=True)
        
        queryEmbbert = self.bert(query_bert, query_bertmask)
        queryEmbbert1 = queryEmbbert[0][:, 0, :] 
        queryEmbbert2 = queryEmbbert[2][1][:, 0, :]

        queryEmb = self.embFusion(queryEmb1, queryEmb2, queryEmbbert1, queryEmbbert2)
        queryEmb = F.normalize(queryEmb, dim=-1)
        # queryEmb = queryEmb + queryEmbbert
        
        #queryEmb = torch.relu(self.qd1(queryEmb))
        return queryEmb
        
    def docTower(self, input):
        doc, docchar, doc_bert, doc_bertmask = input
        mask = torch.gt(doc, 0).unsqueeze(dim=-1).float()
        docwordEmb = self.wordemb(doc)
        docwordEmb = self.word2char(docwordEmb)
        doccharemb = self.charemb(docchar)
        positionEmb = self.position_emb(self.position_ids[:, :doc.shape[1]])
        #docEmb = torch.sum(docEmb * mask, dim=-2)
        #docEmb = docEmb / torch.sum(mask, dim=-2)
        docEmb = docwordEmb + doccharemb 
        # docEmb = docEmb.permute(0, 2, 1)
        # docEmb = self.conv1d(docEmb)
        # docEmb = self.conv1d2(docEmb)
        # docEmb = docEmb.permute(0, 2, 1)
        # docEmb = self.docatt(docEmb, mask)
        # docEmb = self.dd1(docEmb)
        docEmb1 = self.block((docEmb, mask)) 
        docEmb2 = torch.sum(docEmb * mask, dim=1) / torch.sum(mask.squeeze(), dim=1, keepdim=True)
        
        docEmbbert = self.bert(doc_bert, doc_bertmask)
        docEmbbert1 = docEmbbert[0][:, 0, :] 
        docEmbbert2 = docEmbbert[2][1][:, 0, :]
        
        docEmb = self.embFusion(docEmb1, docEmb2, docEmbbert1, docEmbbert2)
        # docEmb = docEmb + docEmbbert
        
        docEmb = = F.normalize(docEmb, dim=-1)
        #docEmb = torch.relu(self.dd1(docEmb))
        return docEmb
        
    def forward(self, input):
        if not self.harduse:
            query, querychar, doc, docchar, neg, negchar,  queries_bert, docs_bert, negs_bert, queries_bertmask, docs_bertmask, negs_bertmask = input
        else:
            query, querychar, doc, docchar, neg, negchar,  queries_bert, docs_bert, negs_bert, queries_bertmask, docs_bertmask, negs_bertmask, hardneg, hardneg_char, hard_bert, hard_bertmask = input
        # maxlen = max(doc.shape[1], neg.shape[1])
        # doc = F.pad(doc, (0, maxlen-doc.shape[1]))
        # docchar = F.pad(docchar, (0, maxlen-docchar.shape[1]))
        # neg = F.pad(neg, (0, maxlen-neg.shape[1]))
        # negchar = F.pad(negchar, (0, maxlen-negchar.shape[1]))
        queryEmb = self.queryTower((query, querychar, queries_bert, queries_bertmask))
        docEmb = self.queryTower((doc, docchar, docs_bert, docs_bertmask))
        negEmb = self.queryTower((neg, negchar, negs_bert, negs_bertmask))
        output = (queryEmb, docEmb, negEmb)
        # reshape
        if self.harduse:
            b, hard_num, s = hardneg.shape
            hardneg = hardneg.reshape(-1, s)
            hardneg_char = hardneg_char.reshape(-1, s)
            hard_bert = hard_bert.reshape(-1, hard_bert.shape[-1])
            hard_bertmask = hard_bertmask.reshape(-1, hard_bertmask.shape[-1])
            hardnegEmb = self.docTower((hardneg, hardneg_char, hard_bert, hard_bertmask))
            hardnegEmb = hardnegEmb.reshape(b, hard_num, -1)
            output += (hardnegEmb, )
        loss = self.loss(output)
        return loss
    
    def loss(self, input):
        if not self.harduse:
            queryEmb, docEmb, negEmb = input
        else:
            queryEmb, docEmb, negEmb, hardnegEmb = input
        docAndNegEmb = torch.cat([docEmb, negEmb], dim=0)
        scores = torch.matmul(queryEmb, docAndNegEmb.t())   # (batch_size, batch_size+neg_nums)
        if self.harduse:
            scores_hard = self.hardscore(queryEmb, hardnegEmb)
            scores = torch.cat([scores, scores_hard], dim=-1)  # batch_size, batch_size+neg_num+hard_negnums
        labels = torch.arange(scores.shape[0]).cuda()
        # loss1 = self.cretirion(scores, labels)
        # scores = scores[:, :scores.shape[0]].t()
        # loss2 = self.cretirion(scores, labels)
        pos_score = torch.diag(scores[:, :scores.shape[0]])  # batch_size
        scores_trans = (scores - pos_score.unsqueeze(dim=-1))*20  # 强制分开
        loss = torch.logsumexp(scores_trans, dim=-1).mean()
        # return loss1 + loss2 * 0.4 
        return loss
    
    def hardscore(self, queryEmb, hardnegEmb):
        b = queryEmb.shape[0]
        scores = queryEmb.unsqueeze(dim=1) * hardnegEmb   # b, neg_num, dim
        scores = torch.sum(scores, dim=-1)  # b, neg_num
        # pos_scores = scores[:, 0]
        # neg_scores = scores[:, 1:]
        # pos_scores_aux = 0.7 - pos_scores  # 希望pos_scores的分数大于0.7
        # pos_scores_aux = pos_scores_aux.unsqueeze(dim=1)
        # pos_scores = pos_scores.view(-1, 1).expand_as(neg_scores)  # b, neg_num
        # loss = torch.cat([torch.zeros(b, 1).cuda(), neg_scores - pos_scores, pos_scores_aux], dim=-1) * 10  # b, 1+neg_num
        # loss = torch.logsumexp(loss, dim=-1)  # b
        return scores
          
    def scores(self, input):
        if not self.harduse:
            query, querychar, doc, docchar, neg, negchar, queries_bert, docs_bert, negs_bert, queries_bertmask, docs_bertmask, negs_bertmask = input
        else:
            query, querychar, doc, docchar, neg, negchar, queries_bert, docs_bert, negs_bert, queries_bertmask, docs_bertmask, negs_bertmask, hardneg, hardnegchar, hard_bert, hard_bertmask = input
        
        queryEmb = self.queryTower((query, querychar, queries_bert, queries_bertmask))
        # maxlen = max(doc.shape[1], neg.shape[1])
        # doc = F.pad(doc, (0, maxlen-doc.shape[1]))
        # docchar = F.pad(docchar, (0, maxlen-docchar.shape[1]))
        # neg = F.pad(neg, (0, maxlen-neg.shape[1]))
        # negchar = F.pad(negchar, (0, maxlen-negchar.shape[1]))
        docEmb = self.docTower((doc, docchar, docs_bert, docs_bertmask))
        negEmb = self.docTower((neg, negchar, negs_bert, negs_bertmask))
        docAndNegEmb = torch.cat([docEmb, negEmb], dim=0)
        scores = torch.matmul(queryEmb, docAndNegEmb.t())    # (batch_size, batch_size+negnums)
        if self.harduse:
            b, hard_num, s = hardneg.shape
            hardneg = hardneg.reshape(-1, s)
            hardnegchar = hardnegchar.reshape(-1, s)
            hard_bert = hard_bert.reshape(-1, hard_bert.shape[-1])
            hard_bertmask = hard_bertmask.reshape(-1, hard_bertmask.shape[-1])
            hardnegEmb = self.docTower((hardneg, hardnegchar, hard_bert, hard_bertmask))
            hardnegEmb = hardnegEmb.reshape(b, hard_num, -1)
            scoreshard = torch.sum(queryEmb.unsqueeze(dim=1) * hardnegEmb, dim=-1)  # batch_size, hardnegnums
            scores = torch.cat([scores, scoreshard], dim=-1)  # batch_size, batch_size+negnum+hardnegnum
        
        _, indices = scores.topk(5)   # (b, 10)
        label = torch.arange(queryEmb.shape[0]).view(-1, 1).expand_as(indices).cuda()
        hits = (label == indices).nonzero() 
        if  len(hits) == 0:
            return 0, 0
        else:
            hits = (hits[:, 1] + 1).float()
            hits = torch.reciprocal(hits)
            hits5 = torch.sum(hits)
        indices = scores.argmax(dim=-1)
        hits1 = torch.sum(torch.eq(indices, torch.arange(queryEmb.shape[0]).cuda()))
        return hits5.item(), hits1.item()
    
    def positionalencoding1d(self, d_model, length):
        """
            :param d_model: dimension of the model
            :param length: length of positions
            :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                            -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.position_emb.weight.data = pe
        self.position_emb.weight.requires_grad = False

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
                 kernel_size=2,
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
            padding=0
        )
        if self.skip_connect and self.in_channels != self.out_channels:
            self.conv1d_1x1 = nn.Conv1d(self.in_channels, self.out_channels, padding=0)
        if self.drop_gate:
            self.dropout = nn.Dropout(drop_gate)
    def forward(self, input):
        xo, mask = input
        #mask = torch.gt(xo, 0).unsqueeze(dim=-1).float()
        x = xo * mask 
        x = self.conv1d(x)  
        x, g = x[:, :self.out_channels, ...], x[:, self.out_channels:, ...]
        if self.drop_gate:
            g = self.dropout(g)
        g = F.sigmoid(g)
        if self.skip_connect:
            if self.in_channels != self.out_channels:
                xo = self.conv1d_1x1(xo)
            b, d, s = xo.shape
            x = torch.cat([x, torch.zeros(b, d, s-x.shape[-1]).cuda()], dim=-1)
            g = torch.cat([g, torch.zeros(b, d, s-g.shape[-1]).cuda()], dim=-1)
            return xo * (1 - g) + x * g
        else:
            return x * g * mask
    
class Block(nn.Module):
    def __init__(self, in_features=128) -> None:
        super().__init__()
        # self.conv1d1 = DilatedGatedConv1D(in_features, in_features, dilation=1, drop_gate=0.1)
        # self.conv1d2 = DilatedGatedConv1D(in_features, in_features, dilation=2, drop_gate=0.1)
        # self.conv1d3 = DilatedGatedConv1D(in_features, in_features, dilation=1, drop_gate=0.1)
        self.conv1d1 = self.conv1d = nn.Conv1d(in_features, in_features, kernel_size=3, dilation=1, padding=1) 
        self.att = AttentionPooling1D(in_features) 
    def forward(self, input):
        x0, mask = input
        x0 = x0.permute(0, 2, 1)
        mask = mask.permute(0, 2, 1)
        # x0 = self.conv1d1((x0, mask))
        # x0 = self.conv1d2((x0, mask))
        # x0 = self.conv1d3((x0, mask))
        x0 = self.conv1d1(x0)
        x0 = x0.permute(0, 2, 1) # 
        mask = mask.permute(0, 2, 1)
        x0 = self.att(x0, mask)
        return x0  # b*emd
        