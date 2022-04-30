# aliwentian
很多品牌词比较重要, 品牌匹配
v1: 传统匹配
v2: 基于bert

1. query: char embedding + word embedding + mean_pooling + dense 
   doc: word embedding + mean_pooling
   all data mrr@10: 0.039

2. query: char embedding + word embedding + mean_pooling + dense
   doc: word embedding + attention 
   all data mrr@10: 0.044

3. query: char embedding + word embedding + attention + dense
   doc: word embedding + attention 
   all data mrr@10: 0.050
   线上: 0.07

4. query: char embedding + word embedding + block(conv*3 + attention)
   doc: word embedding + block(conv + attention)
   不如3

5. query、doc都只是用word embedding
   简化block结构(conv*1 + attention)
   繁体转简体，全角转半角

6. 传统模型分数到过mrr@10: 0.147 

7. 类似lebert, 在0层token embedding 融合 word embedding, 收敛上是更快的, 有没有没有对比验证, 单估计不会导致降低

8. 没有尝试过预训练，最后分数0.251，a榜最高0.42， 分数不理想，不过学习到了一些知识吧

9. 所有层cls向量简单加权融合, 权重为可训练的参数
   效果变差

10. 训练好后, 加入hard sample, 微调最后几层，有提升







