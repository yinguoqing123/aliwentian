from segword import BIMMSegment
import gensim
import numpy as np
import json


path = '../word2vec/word2vec.bin'
word2vec = gensim.models.KeyedVectors.load(path)
seg = BIMMSegment(word2vec.index_to_key[:1500000])

def cut(s):
    #s = ''.join([' ' + char + ' ' if self.isDigitOrAlpha(char) else char for char in s])
    s = s.lower()
    words = []
    for sub in s.split():
        sub = seg.lcut(sub)
        words.extend(sub)
    return words

key_words = set(word2vec.index_to_key[:1500000])
def getKeyWords(words):
    keys = []
    default_vec = np.zeros(100)
    words_vec = []
    for word in words:
        if word in key_words:
            words_vec.append(word2vec.get_vector(word))
        else:
            words_vec.append(default_vec)
    words_vec = np.array(words_vec)
    p = np.dot(words_vec, words_vec.T)
    np.fill_diagonal(p, 0)
    p = p.sum(axis=1)
    ids = p.argsort()
    keys_cnt = round(len(words) * 0.25)
    if keys_cnt < 4:
        keys = [words[i] for i in ids][-4:]
    elif keys_cnt >= 6:
        keys = [words[i] for i in ids][-6:]
    else:
        keys = [words[i] for i in ids][-keys_cnt:]
    return keys
            
f = open('../data/corpus_keywords.tsv', 'w', encoding='utf-8') 
i = 1

with open("../data/corpus.tsv", 'r', encoding='utf-8') as ff:
    lines = ff.readlines()
    for line in lines:
        line = line.strip().split('\t')[1]
        words = cut(line)
        keys = getKeyWords(words)
        keys = json.dumps(keys, ensure_ascii=False)
        f.write(f"{str(i)}\t{keys}\n")
        i += 1
        
        