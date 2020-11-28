import pandas as pd
import numpy as np
import random
import math
import re
import json
import string
import torch
import pdb
exclude = set(string.punctuation)
exclude2 = set([',','.','?','"','!'])


def f1_metric(target, prediction):
    f1 = []
    for y_t, y_p in zip(target, prediction):
        if len(set(y_p)) == 0 and len(set(y_t)) == 0:
            f1.append(1.)
        elif len(set(y_p)) == 0 and len(set(y_t)) != 0:
            f1.append(0.)
        elif len(set(y_t)) == 0:
            f1.append(0.)                
        else:
            P = 1. * len(set(y_t) & set(y_p))/len(set(y_p))
            R = 1. * len(set(y_t) & set(y_p))/len(set(y_t))
            if P + R == 0:
                f1.append(0.)
            else:
                f1.append(2.*P*R / (P + R))
    return np.mean(f1)
    
def split_dev(data, frac=0.1):
    dev_len = int(frac*len(data))
    indices = list(range(len(data)))
    random.shuffle(indices)
    dev_ind, train_ind = indices[:dev_len], indices[dev_len:]
    return data.iloc[train_ind], data.iloc[dev_ind]

def span2target(span, text):
    tokens = re.split(' |\n',text)
    target = [0]*len(tokens)
    span_list = json.loads(span)
    if len(span_list) == 0:
        return target
    word_label = []
    for i, t in enumerate(tokens):
        if i > 0:
            word_label.append(-1)
        for ch in t:
            word_label.append(i)
    assert word_label[-1]+1 == len(tokens)
    try:
        ws_set = set([word_label[i] for i in span_list])
    except:
        pdb.set_trace()
    if -1 in ws_set:
        ws_set.remove(-1)
    word_span = sorted(list(ws_set))
    for word_id in word_span:
        target[word_id] = 1
    return target

def target2span(target, text):
    tokens = re.split(' |\n',text)
    target_ind = []
    for i,t in enumerate(target):
        if t==1:
            target_ind.append(i)
    if len(target_ind) == 0:
        return target_ind
    word_label = []
    char_label = []
    j = 0
    for i, t in enumerate(tokens):
        if i > 0:
            word_label.append(-1)
            k = 1
            while text[j-k] in exclude2 and j-k >= 0:
                word_label[-1-k] = -1
                k += 1
            char_label.append(j)
            j+=1
        for ch in t:
            if ch in exclude:
                word_label.append(-1)
            else:
                word_label.append(i)
            char_label.append(j)
            j+=1
    k = 1
    while text[-k] in exclude2 and k>=0:
        word_label[-k] = -1
        k += 1
    spans = []
    for i, wl in enumerate(word_label):
        if wl in target_ind:
            spans.append(i)
    spans_ = []
    prev = -100
    for i in spans:
        if i - prev == 2:
            spans_.append(i-1)
        spans_.append(i)
        prev = i
    return spans_

class Dictionary(object):
    def __init__(self, path=''):
        self.word2idx = dict()
        self.char2idx = dict()
        self.idx2word = list()
        self.idx2char = list()
        chars = list(string.ascii_lowercase)+[str(i) for i in range(10)]+['*','<','>','?']
        for i, ch in enumerate(chars):
            self.idx2char.append(ch)
            self.char2idx[ch] = i
        if path != '':
            words = json.loads(open(path, 'r').readline())
            for item in words:
                self.add_word(item)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def cdict_len(self):
        return len(self.idx2char)

    def __len__(self):
        return len(self.idx2word)

def clean_str(string):
    lst = re.split(' |\n',string)
    cstr = []
    for token in lst:
        token = token.lower()
        cleaned = re.sub("[^a-z0-9*]", "", token)
        if cleaned == "":
            cleaned = '<pad>'
        cstr.append(cleaned)
    return cstr

class DataIns(object):
    def __init__(self, df, dictionary):
        self.raw_text = df['text']
        self.spans = df['spans']
        self.dictionary = dictionary
        self.target = span2target(df['spans'], df['text'])
        self.data_w, self.data_c = self._pack()
    
    def _pack(self):
        cleaned = clean_str(self.raw_text)
        word_enc = [self.dictionary.word2idx[y] for y in cleaned]
        char_enc = []
        for tk in cleaned:
            char_enc.append([self.dictionary.char2idx[ch] for ch in tk])
        return word_enc, char_enc

    def __len__(self):
        return len(self.data_w)

class DataSet(object):
    def __init__(self, df, dictionary, is_train=True):
        self._data_raw = []
        self.data_list = []
        self.dictionary = dictionary
        for i, row in df.iterrows():
            di = DataIns(row, dictionary)
            self.data_list.append(di)
            self._data_raw.append({'text':di.raw_text, 'spans':json.loads(di.spans)})
        self._data_w, self._data_c, self._target, self._mask = self._pack(is_train=is_train)

    def get_tensor(self):
        return self._data_w, self._data_c, self._target, self._mask
        
    def _pack(self, is_train=True):
        max_wlen = max([len(x) for x in self.data_list])
        max_clen = -1
        for x in self.data_list:
            max_clen = max(max([len(c) for c in x.data_c]), max_clen)
        mask = []
        targets = []
        seqs = []
        chars = []
        for di in self.data_list:
            mask.append([1]*len(di)+[0]*(max_wlen - len(di)))
            targets.append(di.target + [-1]*(max_wlen - len(di)))
            seqs.append(di.data_w + [self.dictionary.word2idx['<pad>'] for _ in range(max_wlen - len(di))])
            char_t = [] 
            for c_list in di.data_c:
                char_t.append(c_list + [self.dictionary.char2idx['?'] for _ in range(max_clen - len(c_list))])
            for _ in range(max_wlen - len(di.data_c)):
                char_t.append([self.dictionary.char2idx['?'] for _ in range(max_clen)])
            chars.append(char_t)
        with torch.set_grad_enabled(is_train):
            se, ce, ta, ma = torch.tensor(seqs, dtype=torch.long).t(), torch.tensor(chars, dtype=torch.long).permute(2,1,0).contiguous(), torch.tensor(targets, dtype=torch.long), torch.tensor(mask, dtype=torch.long)
        return se, ce, ta, ma

    def __getitem__(self, index):
        return self._data_raw[index]

    def __len__(self):
        return len(self._data_raw)    

    def seq_len(self, index):
        return len(self.data_list[index])
            
