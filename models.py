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

class NoAttention(nn.Module):
    def __init__(self):
        super(NoAttention, self).__init__()
        pass

    def forward(self, input, input_raw=None, mask=None):
        return torch.mean(input, dim=0, keepdim=False)

class Attention(NoAttention):
    def __init__(self, inp_size, attention_unit, attention_hops, dictionary, dropout):
        super(Attention,self).__init__()
        self.ws1 = nn.Linear(inp_size, attention_unit, bias=False)
        self.ws2 = nn.Linear(attention_unit, attention_hops, bias=False)
        self.dictionary = dictionary
        self.attention_hops = attention_hops
        self.drop = nn.Dropout(dropout)

    def get_mask(self,input_raw):
        transformed_inp = torch.transpose(input_raw, 0, 1).contiguous()  # [bsz, seq_len]
        transformed_inp = transformed_inp.view(input_raw.size()[1], 1, input_raw.size()[0])  # [bsz, 1, seq_len]
        concatenated_inp = [transformed_inp for i in range(self.attention_hops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, seq_len]
        mask = (concatenated_inp == self.dictionary.char2idx['?']).float()
        mask = mask[:,:,:input_raw.size(0)]
        return mask

    def forward(self, input, input_raw=None, mask=None): # input --> (seq_len, bsize, inp_size) input_raw --> (seq_len, bsize)
        inp = torch.transpose(input, 0, 1).contiguous()
        size = inp.size()  # [bsz, seq_len, inp_size]
        compressed_embeddings = inp.view(-1, size[2])  # [bsz*seq_len, inp_size]
        if input_raw is not None:
            mask = self.get_mask(input_raw) # need this to mask out the <pad>s
        else:
            assert mask is not None
        hbar = torch.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*seq_len, attention-unit]
        alphas = self.ws2(self.drop(hbar)).view(size[0], size[1], -1)  # [bsz, seq_len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, seq_len]
        penalized_alphas = alphas + -10000*mask
        alphas = F.softmax(penalized_alphas.view(-1, size[1]),1)  # [bsz*hop, seq_len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, seq_len]
        out_agg, attention = torch.bmm(alphas, inp), alphas # [bsz, hop, inp_size], [bsz, hop, seq_len] 
        return out_agg.squeeze(1)

class CharEncoder(nn.Module):
    def __init__(self,config):
        super(CharEncoder, self).__init__()
        self.emb = Embed(config['dictionary'].cdict_len(), config['dictionary'], int(config['ninp']/2), word_vector = None)
        self.rnn = RNN(int(config['ninp']/2), int(config['nhid']/2), config['nlayers'], config['dropout'])
        if config['attention']:
            self.attention = Attention(config['nhid'], config['attention-unit'], 1, config['dictionary'], config['dropout'])
        else:
            self.attention = NoAttention()
        self.drop = nn.Dropout(config['dropout'])

    def init_hidden(self,bsz):
        return self.rnn.init_hidden(bsz)

    def forward(self, input, hidden): # char_len, word_len, bsz --> word_len, bsz, hidd_size
        emb_out = self.emb(input) # char_len, word_len, bsz, emb_size
        char_len, word_len, bsz, emb_size = emb_out.size()
        emb_out = emb_out.view(char_len, -1, emb_size)
        rnn_out = self.rnn(emb_out, hidden)
        out_agg = self.attention(rnn_out,input.view(char_len, -1))
        return out_agg.view(word_len, bsz, -1)

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
        self.char_emb = CharEncoder(config)
        self.rnn = RNN(2 * config['ninp'], config['nhid'], config['nlayers'], config['dropout'])# 2*
        self.classifier = Classifier(2 * config['nhid'], config['nclasses'], config['dropout'])
    
    def init_hidden(self, bsz):
        return self.rnn.init_hidden(bsz)

    def forward(self, input_w, input_c, hidden, mask):
        emb_word = self.emb(input_w)
        hidden_c = self.char_emb.init_hidden(input_c.size(1)*input_c.size(2))
        emb_char = self.char_emb(input_c, hidden_c)
        emb_out = torch.cat([emb_word,emb_char], dim=2)
        rnn_out = self.rnn(emb_out, hidden) #emb_out
        scores = self.classifier(rnn_out) #seq_len, bsz, nclasses
        return scores.permute(1,0,2).contiguous() #bsz, seq_len, nclasses
