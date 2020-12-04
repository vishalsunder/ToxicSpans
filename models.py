import pdb
import numpy as np
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torchcrf import CRF
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
                        #print("uncached word: " + word)
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

class CNN(nn.Module):
    def __init__(self, num_filters, filter_height, dropout):
        super(CNN,self).__init__()
        self.num_filters = num_filters
        self.conv = nn.Conv2d(1,num_filters,(filter_height,1),stride=1) #nn.ModuleList([nn.Conv2d(1,num_filters,(filter_height,i),stride=1) for i in filter_wts])
        self.drop = nn.Dropout(dropout)

    def forward(self,input): # char_len, word_len*bsz, emb_size
        seq_len = input.size(0)
        bsz = input.size(1)
        emb = input.size(2)
        inp = input.permute(1,2,0).unsqueeze(1)
        cout = self.conv(self.drop(inp)) # (bsz, nfilter, 1, seq_len)
        out_cnn = F.max_pool2d(cout,(1,seq_len)).squeeze(3).squeeze(2) #bsz,nfilter
        return out_cnn

class NoAttention(nn.Module):
    def __init__(self):
        super(NoAttention, self).__init__()
        pass

    def forward(self, input, input_raw=None, mask=None):
        return torch.mean(input, dim=0, keepdim=False)

class Attention(NoAttention):
    def __init__(self, inp_size, attention_hops, dropout):
        super(Attention,self).__init__()
        self.Wq = nn.ModuleList()
        self.Wk = nn.ModuleList()
        self.Wv = nn.ModuleList()
        for i in range(attention_hops):
            self.Wq.append(nn.Linear(inp_size,int(inp_size/attention_hops),bias=False))
            self.Wk.append(nn.Linear(inp_size,int(inp_size/attention_hops),bias=False))
            self.Wv.append(nn.Linear(inp_size,int(inp_size/attention_hops),bias=False))
        self.drop = nn.Dropout(dropout)

    def forward(self, input, mask=None):
        out = []
        for wq, wk, wv in zip(self.Wq, self.Wk, self.Wv):
            query = wq(self.drop(input))
            key = wk(self.drop(input))
            value = wv(self.drop(input))
            mask_ = 1. - torch.cat([mask.unsqueeze(1) for _ in range(input.size(0))], dim=1)
            align = torch.tanh(torch.bmm(query.permute(1,0,2), key.permute(1,2,0))/(value.size(2))**0.5) - 10000*mask_ # bsz, seq_len, seq_len
            align = torch.softmax(align, dim=2)
            out.append(torch.bmm(align, value.permute(1,0,2)).permute(1,0,2))
        output = torch.cat(out, dim=2)
        return output

class CharEncoderCnn(nn.Module):
    def __init__(self,config):
        super(CharEncoderCnn, self).__init__()
        self.emb = Embed(config['dictionary'].cdict_len(), config['dictionary'], config['ninp'], word_vector = None)
        self.cnn = CNN(config['ninp'], config['nhid'], config['dropout'])
        self.drop = nn.Dropout(config['dropout'])

    def forward(self, input): # char_len, word_len, bsz --> word_len, bsz, hidd_size
        emb_out = self.emb(input) # char_len, word_len, bsz, emb_size
        char_len, word_len, bsz, emb_size = emb_out.size()
        emb_out = emb_out.view(char_len, -1, emb_size)
        cnn_out = self.cnn(emb_out)
        return cnn_out.view(word_len, bsz, -1)

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
        self.char_emb = CharEncoderCnn(config)
        self.rnn = RNN(2 * config['ninp'], config['nhid'], config['nlayers'], config['dropout'])# 2*
        self.classifier = Classifier(2 * config['nhid'], config['nclasses'], config['dropout'])
        if config['attention']:
            self.attention = Attention(2 * config['nhid'], config['num-heads'], config['dropout'])
        else:
            self.attention = NoAttention()           
    
    def init_hidden(self, bsz):
        return self.rnn.init_hidden(bsz)

    def forward(self, input_w, input_c, hidden, mask):
        emb_word = self.emb(input_w)
        #hidden_c = self.char_emb.init_hidden(input_c.size(1)*input_c.size(2))
        emb_char = self.char_emb(input_c)#, hidden_c)
        emb_out = torch.cat([emb_word,emb_char], dim=2)
        rnn_out = self.rnn(emb_out, hidden) #emb_out
        att_out = self.attention(rnn_out, mask=mask)
        scores = self.classifier(rnn_out) #seq_len, bsz, nclasses
        return scores.permute(1,0,2).contiguous() #bsz, seq_len, nclasses
