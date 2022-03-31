import torch
from torch import nn
from torch.nn import functional as F
import jieba
import gensim

import os

"""
    Trie：前缀字典树
    insert：添加字符串
    search：查找字符串
    start_with: 查找字符前缀
    get_freq: 获取词频
"""

class Trie:
    def __init__(self):
        self.root = {}
        self.max_word_len = 0
        self.total_word_freq = 0
        self.end_token = '[END]'
        self.freq_token = '[FREQ]'

    def insert(self, word: str, freq: int = 1, tag: str = None):
        node = self.root
        for char in word:
            node = node.setdefault(char, {})
        self.max_word_len = max(self.max_word_len, len(word))
        node[self.end_token] = self.end_token
        node[self.freq_token] = freq
        self.total_word_freq += freq

    def search(self, word: str):
        node = self.root
        for char in word:
            if char not in node:
                return None
            node = node[char]
        return node if self.end_token in node else None

    def start_with(self, prefix: str):
        node = self.root
        for char in prefix:
            if char not in node:
                return None
            node = node[char]
        return node

    def get_freq(self, word: str):
        node = self.search(word)
        if node:
            return node.get(self.freq_token, 1)
        else:
            return 0

class BaseSegment:
    
    trie = Trie()
    
    def cut(self, sentence: str):
        pass
    
    @classmethod
    def load_dict(cls, d: str):
        for word in d:
            cls.trie.insert(word)
        print('词典加载完成！')

class FMMSegment(BaseSegment):
    def __init__(self):
        super(FMMSegment, self).__init__()

    def cut(self, sentence: str):
        if sentence is None or len(sentence) == 0:
            return []

        index = 0
        text_size = len(sentence)
        while text_size > index:
            word = ''
            for size in range(self.trie.max_word_len+index, index, -1):
                word = sentence[index:size]
                if self.trie.search(word):
                    index = size - 1
                    break
            index = index + 1
            yield word


class RMMSegment(BaseSegment):
    def __init__(self):
        super(RMMSegment, self).__init__()

    def cut(self, sentence: str):
        if sentence is None or len(sentence) == 0:
            return []

        result = []
        index = len(sentence)
        window_size = min(index, self.trie.max_word_len)
        while index > 0:
            word = ''
            for size in range(index-window_size, index):
                word = sentence[size:index]
                if self.trie.search(word):
                    index = size + 1
                    break
            index = index - 1
            result.append(word)
        result.reverse()
        for word in result:
            yield word

class BIMMSegment(BaseSegment):
    def __init__(self, d):
        super(BIMMSegment, self).__init__()
        self.FMM = FMMSegment()
        self.RMM = RMMSegment()
        self.load_dict(d)

    def cut(self, sentence: str):
        if sentence is None or len(sentence) == 0:
            return []
        res_fmm = [word for word in self.FMM.cut(sentence)]
        res_rmm = [word for word in self.RMM.cut(sentence)]
        if len(res_fmm) == len(res_rmm):
            if res_fmm == res_rmm:
                result = res_fmm
            else:
                f_word_count = len([w for w in res_fmm if len(w) == 1])
                r_word_count = len([w for w in res_rmm if len(w) == 1])
                result = res_fmm if f_word_count < r_word_count else res_rmm
        else:
            result = res_fmm if len(res_fmm) < len(res_rmm) else res_rmm
        for word in result:
            yield word
            
    def lcut(self, sentence):
        return list(self.cut(sentence))
       
if __name__ == '__main__':     
    path = '../word2vec/word2vec.bin'
    word2vec = gensim.models.KeyedVectors.load(path)
    d = word2vec.index_to_key[:1500000]
    text = '男鞋2021新款秋季无后跟半拖休闲皮鞋男士一脚蹬韩版懒人豆豆鞋男'
    segment = BIMMSegment(d)
    print(' '.join(segment.cut(text)))