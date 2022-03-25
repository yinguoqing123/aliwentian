import json

charCount = {}

with open("../data/corpus.tsv", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split('\t')[1]
        for char in line:
            if char != ' ':
                charCount[char] = charCount.get(char, 0) + 1

with open("../data/train.query.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split('\t')[1]
        for char in line:
            if char != ' ':
                charCount[char] = charCount.get(char, 0) + 1
            
with open("../data/dev.query.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split('\t')[1]
        for char in line:
            if char != ' ':
                charCount[char] = charCount.get(char, 0) + 1
            

charfreq = list(charCount.items())
charfreq.sort(key=lambda x: x[1], reverse=True)

char2id = {'<pad>': 0, '<unk>': 1}
for char, freq in charfreq:
    if freq > 5:
        char2id[char] = len(char2id)   # 0: <pad>  1: <unk>     

id2char = {j: i for i, j in char2id.items()}        

json.dump((char2id, id2char), open('charidmap.json', 'w'))