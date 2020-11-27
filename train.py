from models import *
from transformers import BertTokenizer, DistilBertTokenizer

from utils import *

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import json
import time
from time import sleep
import sys
import random
import os
import pdb

from sklearn.metrics import f1_score 

class Trainer:
    def __init__(self, data, dictionary, device, args, criterion = None, optimizer = None):
        self.data = data
        self.dictionary = dictionary
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.args = args 

    def update_opt(self, optimizer):
        self.optimizer = optimizer

    def data_shuffle(self):
        self.data = self.data.sample(frac=1).reset_index(drop=True)

    #def bertpackage(self, data, is_train = True):
    #    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    #    data = [json.loads(x) for x in data]
    #    dat = tokenizer([" ".join(x['text']) for x in data], return_tensors="pt", padding=True, truncation=True)
    #    targets = [x['label'] for x in data]
    #    with torch.set_grad_enabled(is_train):
    #        targets = torch.tensor(targets, dtype=torch.long)
    #    return dat, targets

    #def bt_on_device(self, tokenizer):
    #    t2 = {}
    #    for key, val in tokenizer.items():
    #        t2[key] = val.to(self.device)
    #    return t2

    def opt_step(self, model, loss):
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), self.args.clip)
        self.optimizer.step()

    def get_span(self, prediction, mask, data):
        output = []
        for i, pred_tens in enumerate(prediction):
            output.append(target2span(pred_tens[mask[i] != 0].tolist(), data[i]['text']))
        return output

    def evaluate(self, model, data_val, bsz = 32):
        model.eval()
        total_correct = 0
        y_pred = []
        y_true = []
        preds = 0
        for batch, i in enumerate(range(0, len(data_val), bsz)):
            sys.stdout.write('\r')
            j=(batch+1) / (len(list(range(0, len(data_val), bsz))))
            last = min(len(data_val), i+bsz)
            indata = data_val[i:last]
            data_batch = DataSet(indata, self.dictionary, is_train=False)
            data, targets, mask = data_batch.get_tensor()
            data, targets, mask = data.to(self.device), targets.to(self.device), mask.to(self.device)
            hidden = model.init_hidden(bsz)
            output = model.forward(data, hidden)
            prediction = self.get_span(torch.max(output, dim=2)[1].cpu(), mask.cpu(), data_batch) # bsz, seq_len
            y_pred.extend(prediction)
            y_true.extend([data_batch[i]['spans'] for i in range(len(data_batch))])
            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
            sys.stdout.flush()
            sleep(0)
        print('\n')
        f1 = f1_metric(y_true, y_pred)
        return f1, y_pred, y_true

    def forward(self, i, model, data, bsz=32):
        last = min(len(data), i+bsz)
        indata = data[i:last]
        data_batch = DataSet(indata, self.dictionary)
        data, targets, mask = data_batch.get_tensor() 
        data, targets, mask = data.to(self.device), targets.to(self.device), mask.to(self.device)
        hidden = model.init_hidden(bsz)
        output = model.forward(data, hidden) # output --> bsz, seq_len, nclasses
        output_ = output[mask != 0]
        targets_ = targets[mask != 0]
        loss = self.criterion(output_, targets_)
        return loss

    def epoch(self, ep, model):
        model.train()
        total_loss = []
        for batch, i in enumerate(range(0, len(self.data), self.args.batch_size)):
            sys.stdout.write('\r')
            j=(batch+1) / (len(list(range(0, len(self.data), self.args.batch_size))))
            loss = self.forward(i, model, self.data, self.args.batch_size)
            self.opt_step(model, loss)
            total_loss.append(loss.item())
            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
            sys.stdout.flush()
            sleep(0)
        print('\n')
        return np.mean(total_loss), model 
