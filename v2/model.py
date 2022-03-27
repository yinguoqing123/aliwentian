import torch 
from torch import nn 
import torch.nn.functional as F
import math
from transformers import BertTokenizer, BertModel, BertConfig, AutoConfig, AutoModel, AutoTokenizer


class Model(nn.Module):
    def __init__(self, hardnum=0) -> None:
        super().__init__()
        # self.config = BertConfig.from_pretrained("hfl/chinese-roberta-wwm-ext")  # (attention_probs_dropout_prob, hidden_dropout_prob)
        # self.config.update({'output_hidden_states': True})
        # self.bert = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext", config=self.config)
        #self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        state_dict = torch.load('./pretrained/pytorch_model.bin')
        self.config = AutoConfig.from_pretrained("cyclone/simcse-chinese-roberta-wwm-ext")
        self.config.update({'output_hidden_states': True})
        self.bert = AutoModel.from_pretrained("cyclone/simcse-chinese-roberta-wwm-ext", state_dict=state_dict, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained("cyclone/simcse-chinese-roberta-wwm-ext")
        self.mlp = nn.Linear(768, 128)
        self.cretirion = nn.CrossEntropyLoss()
        self.hardnum = hardnum
        
    def Tower(self, input):
        ids, mask = input
        out = self.bert(ids, mask)
        lastout = torch.sum(out[0], dim=1) / torch.sum(mask, dim=-1, keepdim=True)
        firstout = torch.sum(out[2][0], dim=1) /torch.sum(mask, dim=-1, keepdim=True)
        out =  lastout + firstout
        #out = torch.cat([firstout, lastout], dim=-1)
        out = self.mlp(out)
        out = F.normalize(out, dim=-1)
        return out
    
    # def forward(self, input):
    #     queries, queries_mask, docs, docs_mask, negs, negs_mask = input
    #     queries = self.Tower((queries, queries_mask))
    #     docs = self.Tower((docs, docs_mask))
    #     negs = self.Tower((negs, negs_mask))
    #     docAndNegs = torch.cat([docs, negs], dim=0)
    #     scores = torch.matmul(queries, docAndNegs.t()) * 20  # (batch_size, batch_size+neg_nums)
    #     labels = torch.arange(scores.shape[0]).cuda()
    #     loss1 = self.cretirion(scores, labels)
    #     scores = scores[:, :scores.shape[0]].t()
    #     loss2 = self.cretirion(scores, labels)
    #     pos_scores = torch.sum(queries * docs, dim=-1)
    #     pos_scores_aux = 0.7 - pos_scores  # 希望pos_scores的分数大于0.7
    #     pos_scores_aux = pos_scores_aux.unsqueeze(dim=1)
    #     b = queries.shape[0]
    #     scores = torch.cat([torch.zeros(b, 1).cuda(), pos_scores_aux], dim=-1) * 10  # b, 2
    #     loss = torch.logsumexp(scores, dim=-1)  # b
    #     return loss1 + loss2 * 0.4 + loss.mean()
    
    def forward(self, input):
        if self.hardnum > 0:
            queries, queries_mask, docs, docs_mask, negs, negs_mask, hards, hards_mask = input
        else:
            queries, queries_mask, docs, docs_mask, negs, negs_mask = input
        queries = self.Tower((queries, queries_mask))
        docs = self.Tower((docs, docs_mask))
        negs = self.Tower((negs, negs_mask))
        docAndNegs = torch.cat([docs, negs], dim=0)
        scores = torch.matmul(queries, docAndNegs.t()) * 20 # (batch_size, batch_size+neg_nums)

        if self.hardnum > 0:
            hardnegs = self.Tower((hards, hards_mask))
            hardnegs = hardnegs.reshape(queries.size(0), self.hardnum, -1)
            hard_scores = self.hardscore(queries, hardnegs)
            scores = torch.cat([scores, hard_scores], dim=-1)
        pos_scores = torch.diag(scores[:, :scores.shape[0]])
        scores_trans = scores - pos_scores.unsqueeze(dim=-1)
        loss = torch.logsumexp(scores_trans, dim=-1).mean()
        return loss
    
    def hardscore(self, queryEmb, hardnegEmb):
        b = queryEmb.shape[0]
        scores = queryEmb.unsqueeze(dim=1) * hardnegEmb   # b, neg_num, dim
        scores = torch.sum(scores, dim=-1) * 20  # b, neg_num
        # pos_scores = scores[:, 0]
        # neg_scores = scores[:, 1:]
        # pos_scores_aux = 0.7 - pos_scores  # 希望pos_scores的分数大于0.7
        # pos_scores_aux = pos_scores_aux.unsqueeze(dim=1)
        # pos_scores = pos_scores.view(-1, 1).expand_as(neg_scores)  # b, neg_num
        # loss = torch.cat([torch.zeros(b, 1).cuda(), neg_scores - pos_scores, pos_scores_aux], dim=-1) * 10  # b, 1+neg_num
        # loss = torch.logsumexp(loss, dim=-1)  # b
        return scores
    
    # def forward(self, input):
    #     queries, queries_mask, docs, docs_mask, negs, negs_mask = input
    #     queries1 = self.Tower((queries, queries_mask))
    #     queries2 = self.Tower((queries, queries_mask))
    #     queries = torch.cat([queries1, queries2], dim=0)
    #     docs1 = self.Tower((docs, docs_mask))
    #     docs2 = self.Tower((docs, docs_mask))
    #     docs = torch.cat([docs1, docs2], dim=0)
    #     scores = torch.matmul(queries, docs.t()) * 20  # (batch_size*2, batch_size*2)
    #     b = queries_mask.shape[0]
    #     batch1, batch2, batch3, batch4 = scores[:b, :b],scores[:b, b:], scores[b:, :b], scores[b:, b:]
    #     loss = self.cosloss(batch1) + self.cosloss(batch2) + self.cosloss(batch3) + self.cosloss(batch4)
    #     return loss/4
    
    def cosloss(self, scores):
        pos_score = torch.diag(scores)  # batch_size
        scores_trans = (scores - pos_score.unsqueeze(dim=-1))  # 强制分开
        loss = torch.logsumexp(scores_trans, dim=-1).mean()
        return loss
    
    def scores(self, input):
        if self.hardnum > 0:
            queries, queries_mask, docs, docs_mask, negs, negs_mask, hards, hards_mask = input
        else:
            queries, queries_mask, docs, docs_mask, negs, negs_mask = input
        queries = self.Tower((queries, queries_mask))
        docs = self.Tower((docs, docs_mask))
        negs = self.Tower((negs, negs_mask))
        docAndNegs = torch.cat([docs, negs], dim=0)
        scores = torch.matmul(queries, docAndNegs.t())  # (batch_size, batch_size+neg_nums)
        if self.hardnum > 0:
            hardnegs = self.Tower((hards, hards_mask))
            hardnegs = hardnegs.reshape(queries.size(0), self.hardnum, -1)
            hard_scores = self.hardscore(queries, hardnegs)
            scores = torch.cat([scores, hard_scores], dim=-1)
            
        _, indices = scores.topk(5)   # (b, 10)
        label = torch.arange(queries.shape[0]).view(-1, 1).expand_as(indices).cuda()
        hits = (label == indices).nonzero() 
        if  len(hits) == 0:
            return 0, 0
        else:
            hits = (hits[:, 1] + 1).float()
            hits = torch.reciprocal(hits)
            hits5 = torch.sum(hits)
        indices = scores.argmax(dim=-1)
        hits1 = torch.sum(torch.eq(indices, torch.arange(queries.shape[0]).cuda()))
        return hits5.item(), hits1.item()
    
        