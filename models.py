import pdb
import numpy as np
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from transformers import BertModel, DistilBertModel

class Embed(nn.Module):
    def __init__(self,ntoken, dictionary, ninp, word_vector=None):
        super(Embed, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.dictionary = dictionary
        if word_vector is not None:
            self.encoder.weight.data[self.dictionary.word2idx['<pad>']] = 0
            self.encoder.weight.data[self.dictionary.word2idx['<pad>']] = 0
            if os.path.exists(word_vector):
                print('Loading word vectors from', word_vector)
                vectors = torch.load(word_vector)
                assert vectors[3] >= ninp
                vocab = vectors[1]
                vectors = vectors[2]
                loaded_cnt = 0
                unseen_cnt = 0
                for word in self.dictionary.word2idx:
                    if word not in vocab:
                        to_add = torch.zeros_like(vectors[0]).uniform_(-0.25,0.25)
                        print("uncached word: " + word)
                        unseen_cnt += 1
                        #print(to_add)
                    else:
                        loaded_id = vocab[word]
                        to_add = vectors[loaded_id][:ninp]
                        loaded_cnt += 1
                    real_id = self.dictionary.word2idx[word]
                    self.encoder.weight.data[real_id] = to_add
                print('%d words from external word vectors loaded, %d unseen' % (loaded_cnt, unseen_cnt))  
      
    def forward(self,input):
        return self.encoder(input)


class RNN(nn.Module):
    def __init__(self, inp_size, nhid, nlayers, dropout):
        super(RNN, self).__init__()
        self.nlayers = nlayers
        self.nhid = nhid
        self.rnn = nn.GRU(inp_size, nhid, nlayers, bidirectional=True)
        self.drop = nn.Dropout(dropout)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return torch.zeros(self.nlayers * 2, bsz, self.nhid, dtype=weight.dtype,
                            layout=weight.layout, device=weight.device)

    def forward(self, input, hidden):
        out_rnn = self.rnn(self.drop(input), hidden)[0]
        return out_rnn

class NoAttention(nn.Module):
    def __init__(self):
        super(NoAttention, self).__init__()
        pass

    def forward(self, input):
        return input

class Attention(NoAttention):
    def __init__(self, inp_size, dropout):
        super(Attention, self).__init__()
        self.Wq = nn.Linear(inp_size, inp_size, bias=False)
        self.Wk = nn.Linear(inp_size, inp_size, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, input): # seq_len, bsz, inp_size
        query = self.Wq(self.drop(input))
        key = self.Wk(self.drop(input))
        align = torch.softmax(torch.tanh(torch.bmm(query.permute(1,0,2), key.permute(1,2,0))/(input.size(2))**0.5), dim=2) # bsz, seq_len, seq_len
        return torch.bmm(align, input.permute(1,0,2)).permute(1,0,2)

class Classifier(nn.Module):
    def __init__(self, inp_size, nclasses, dropout):
        super(Classifier, self).__init__()
        self.l1 = nn.Linear(inp_size, inp_size)
        self.l2 = nn.Linear(inp_size, nclasses)
        self.drop = nn.Dropout(dropout)

    def forward(self, input):
        return self.l2(self.drop(torch.tanh(self.l1(self.drop(input)))))

class SpanModel(nn.Module):
    def __init__(self, config):
        super(SpanModel, self).__init__()
        self.emb = Embed(config['ntoken'], config['dictionary'], config['ninp'], config['word-vector'])
        self.rnn = RNN(config['ninp'], config['nhid'], config['nlayers'])
        if config['attention']:
            self.attention = Attention(2 * config['nhid'], config['dropout'])
        else:
            self.attention = NoAttention()
        self.classifier = Classifier(2 * config['nhid'], config['nclasses'], config['dropout'])
    
    def init_hidden(self, bsz):
        return self.rnn.init_hidden(bsz)

    def forward(self, input, hidden):
        emb_out = self.emb(input)
        rnn_out = self.rnn(emb_out, hidden)
        att_out = self.attention(rnn_out)
        scores = self.classifier(att_out) #seq_len, bsz, nclasses
        return scores
