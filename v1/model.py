import torch 
from torch import nn 
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig
from modeling_nezha import NeZhaModel, NeZhaConfig
from lebert import LeBertModel
import math

class Model(nn.Module):
    def __init__(self, nums_word, dims, harduse=False, embedding_init=None) -> None:
        super().__init__()
        # chinese-roberta-wwm-ext
        # self.config = NeZhaConfig.from_pretrained("../pretrained_model/nezha-base-www")
        # self.config.update({'output_hidden_states': True})
        # self.bert = NeZhaModel.from_pretrained('../pretrained_model/nezha-base-www', config=self.config)
        # self.tokenizer = BertTokenizer.from_pretrained('../pretrained_model/nezha-base-www')
        self.config = BertConfig.from_pretrained("hfl/chinese-roberta-wwm-ext")  # (attention_probs_dropout_prob, hidden_dropout_prob)
        self.config.update({'output_hidden_states': True})
        self.bert = LeBertModel.from_pretrained("hfl/chinese-roberta-wwm-ext", config=self.config)
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.wordemb = nn.Embedding(nums_word, 100)
        self.word2bert = nn.Linear(100, 768, bias=False)  # 词向量维度和bert对齐
        self.mlp = nn.Linear(768, dims, bias=False)  # bert输出向量降维
        #self.fuse = nn.Linear(768*2, 2, bias=False)
        self.harduse = harduse
        if embedding_init is not None:
            self.wordemb.weight.data = torch.tensor(embedding_init, dtype=torch.float)
            self.wordemb.weight.requires_grad = False
            
    def fusion(self, input):
        # x = self.fuse(torch.cat(input, dim=-1))
        # weight = torch.softmax(x, dim=-1)
        # out = weight.unsqueeze(dim=-1) * torch.stack(input, dim=1) 
        # out = torch.sum(out, dim=1)
        out = input[0] + input[1]
        return out
        
        
    def queryTower(self, input):
        """
        param: input  
        """
        query_bert, query_bertmask, q_words = input
        q_words = self.wordemb(q_words)
        q_words = self.word2bert(q_words)
        out = self.bert(query_bert, query_bertmask, word_embeddings=q_words)
        queryEmbbert1 = out[0][:, 0, :]
        queryEmbbert2 = out[2][1][:, 0, :]    # CLS 向量
        #queryEmbbert2 = torch.sum(queryEmbbert2, dim=1) / torch.sum(query_bertmask.float(), dim=-1, keepdim=True)  # 1层mean pooling
        queryEmb = self.fusion([queryEmbbert1, queryEmbbert2])
        queryEmb = self.mlp(queryEmb)
        queryEmb = F.normalize(queryEmb, dim=-1)
        
        return queryEmb
        
    def docTower(self, input):
        doc_bert, doc_bertmask, d_words = input

        d_words = self.wordemb(d_words)
        d_words = self.word2bert(d_words)
        out = self.bert(doc_bert, doc_bertmask, word_embeddings=d_words)
        docEmbbert1 = out[0][:, 0, :]
        docEmbbert2 = out[2][1][:, 0, :]  # CLS 向量
        #docEmbbert2 = torch.sum(docEmbbert2, dim=1) / torch.sum(doc_bertmask.float(), dim=-1, keepdim=True)  # 1层mean pooling
        docEmb = self.fusion([docEmbbert1, docEmbbert2])
        docEmb = self.mlp(docEmb)
        docEmb = F.normalize(docEmb, dim=-1)
        return docEmb
    
    def forward(self, input, mode='easy'):
        if mode == 'hard':
            return self.forwardhard(input)
        if mode == 'easy':
            return self.forwardeasy(input)
        if mode == 'pretrain_cse':
            return self.forwardpretrain(input)
        
    def forwardpretrain(self, input):
        # cse 无监督预训练
        docs_bert, docs_bertmask, d_words = input
        docEmb = self.docTower((docs_bert, docs_bertmask, d_words))
        scores = torch.matmul(docEmb, docEmb.t())   # (batch_size, batch_size+neg_nums)
        pos_score = torch.diag(scores)  # batch_size
        scores_trans = (scores - pos_score.unsqueeze(dim=-1))   # 强制分开
        mask = (scores_trans < -0.5).float() * -1e12
        scores_trans = scores_trans*20 + mask 
        scores_trans2 = (scores - pos_score.unsqueeze(dim=0)) 
        mask = (scores_trans2 < -0.5).float() * -1e12
        scores_trans2 = scores_trans2 *20 + mask
        loss = torch.logsumexp(scores_trans, dim=-1).mean() + torch.logsumexp(scores_trans2, dim=0).mean()
        return loss
        
            
    def forwardhard(self, input):
        queries_bert, docs_bert, negs_bert, queries_bertmask, docs_bertmask, negs_bertmask, q_words, d_words, hardneg_words = input
        queryEmb = self.queryTower((queries_bert, queries_bertmask, q_words))
        docEmb = self.docTower((docs_bert, docs_bertmask, d_words))
        negEmb = self.docTower((negs_bert, negs_bertmask, hardneg_words)) 
        pos_scores = torch.sum(queryEmb * docEmb, dim=-1, keepdim=True)
        neg_scores = torch.sum(queryEmb * negEmb, dim=-1, keepdim=True)
        score = torch.cat([pos_scores, neg_scores], dim=-1) - pos_scores
        mask = (score < -0.5).float() * -1e12
        score = score*20 + mask
        loss = torch.logsumexp(score, dim=-1).mean()
        return loss
                
    def forwardeasy(self, input):
        if not self.harduse:
            queries_bert, docs_bert, negs_bert, queries_bertmask, docs_bertmask, negs_bertmask, q_words, d_words, negs_words = input
        else:
            queries_bert, docs_bert, negs_bert, queries_bertmask, docs_bertmask, negs_bertmask, hard_bert, hard_bertmask, q_words, d_words, negs_words, hard_words= input
        # maxlen = max(doc.shape[1], neg.shape[1])
        # doc = F.pad(doc, (0, maxlen-doc.shape[1]))
        # docchar = F.pad(docchar, (0, maxlen-docchar.shape[1]))
        # neg = F.pad(neg, (0, maxlen-neg.shape[1]))
        # negchar = F.pad(negchar, (0, maxlen-negchar.shape[1]))
        queryEmb = self.queryTower((queries_bert, queries_bertmask, q_words))
        docEmb = self.docTower((docs_bert, docs_bertmask, d_words))
        negEmb = self.docTower((negs_bert, negs_bertmask, negs_words))
        output = (queryEmb, docEmb, negEmb)
        # reshape
        if self.harduse:
            b, hard_num, s = hard_bert.shape
            hard_bert = hard_bert.reshape(-1, hard_bert.shape[-1])
            hard_bertmask = hard_bertmask.reshape(-1, hard_bertmask.shape[-1])
            hard_words = hard_words.reshape(-1, hard_words.shape[-1])
            hardnegEmb = self.docTower((hard_bert, hard_bertmask, hard_words))
            hardnegEmb = hardnegEmb.reshape(b, hard_num, -1)
            output += (hardnegEmb, )
        loss, hardids = self.loss(output)
        return loss, hardids
    
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
        pos_score = torch.diag(scores[:, :scores.shape[0]])  # batch_size
        scores_trans = (scores - pos_score.unsqueeze(dim=-1))   # 强制分开
        mask = (scores_trans < -0.3).float() * -1e12
        scores_trans = scores_trans*20 + mask 
        scores_trans2 = (scores[:, :scores.shape[0]] - pos_score.unsqueeze(dim=0)) 
        mask = (scores_trans2 < -0.3).float() * -1e12
        scores_trans2 = scores_trans2 *20 + mask
        loss = torch.logsumexp(scores_trans, dim=-1).mean() + torch.logsumexp(scores_trans2, dim=0).mean()
        hardids = torch.eye(scores.shape[0], scores.shape[1]).cuda() * -1e12  + scores
        _, hardids = hardids.topk(1)
        return loss, hardids
    
    def hardscore(self, queryEmb, hardnegEmb):
        scores = queryEmb.unsqueeze(dim=1) * hardnegEmb   # b, neg_num, dim
        scores = torch.sum(scores, dim=-1)  # b, neg_num
        return scores
          
    def scores(self, input):
        if not self.harduse:
            queries_bert, docs_bert, negs_bert, queries_bertmask, docs_bertmask, negs_bertmask, q_words, d_words, negs_words = input
        else:
            queries_bert, docs_bert, negs_bert, queries_bertmask, docs_bertmask, negs_bertmask, hard_bert, hard_bertmask, q_words, d_words, negs_words, hard_words = input
        
        queryEmb = self.queryTower((queries_bert, queries_bertmask, q_words))
        docEmb = self.docTower((docs_bert, docs_bertmask, d_words))
        negEmb = self.docTower((negs_bert, negs_bertmask, negs_words))
        docAndNegEmb = torch.cat([docEmb, negEmb], dim=0)
        scores = torch.matmul(queryEmb, docAndNegEmb.t())    # (batch_size, batch_size+negnums)
        if self.harduse:
            b, hard_num, s = hard_bert.shape
            hard_bert = hard_bert.reshape(-1, hard_bert.shape[-1])
            hard_bertmask = hard_bertmask.reshape(-1, hard_bertmask.shape[-1])
            hard_words = hard_words.reshape(-1, hard_words.shape[-1])
            hardnegEmb = self.docTower((hard_bert, hard_bertmask, hard_words))
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
