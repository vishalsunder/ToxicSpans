from train import *
from models import *
from utils import *
from sklearn.metrics import f1_score
import pandas as pd
import json
import argparse
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import copy
from torchcrf import CRF

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=300,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers in BiLSTM')
    parser.add_argument('--traindev-data', type=str, default='',
                        help='location of the training data, should be a csv file')
    parser.add_argument('--test-data', type=str, default='',
                        help='location of the test data, should be a csv file')
    parser.add_argument('--valid-data', type=str, default='',
                        help='location of the validation data, should be a csv file')
    parser.add_argument('--attention-unit', type=int, default=350,
                        help='number of attention unit')
    parser.add_argument('--dev-ratio', type=float, default=0.1, help='fraction of train-dev data to use for dev')
    parser.add_argument('--validation', action='store_true', help='use cross validation')
    parser.add_argument('--log', type=str, default='simple.log',
                        help='location to store logs')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='clip to prevent the too large grad in LSTM')
    parser.add_argument('--lr', type=float, default=.001,
                        help='initial learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience value for early stopping')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--stage2', type=int, default=10,
                        help='whether to use stage 2')
    parser.add_argument('--num-heads', type=int, default=2,
                        help='number of attention heads')
    parser.add_argument('--prebert-path', type=str, default='/homes/3/sunder.9/vp-vishal/bert/pretrained_models/',
                        help='path to save the final model')
    parser.add_argument('--dictionary', type=str, default='dict.json',
                        help='path to save the dictionary, for faster corpus loading')
    parser.add_argument('--word-vector', type=str, default='',
                        help='path for pre-trained word vectors (e.g. GloVe), should be a PyTorch model.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--test-bsize', type=int, default=32,
                        help='batch size for testing')
    parser.add_argument('--nclasses', type=int, default=2,
                        help='number of classes')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='type of optimizer')
    parser.add_argument('--attention', action='store_true',
                        help='whether to use dot product attention')
    parser.add_argument('--crf', action='store_true',
                        help='whether to use crf for prediction')

    args = parser.parse_args()
    
    device = torch.device("cpu")
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            device = torch.device("cuda")
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(1111)  # Numpy module.
    random.seed(1111)  # Python random module.
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True

    # Load Dictionary
    assert os.path.exists(args.traindev_data)
    print('Begin to load the dictionary.')
    dictionary = Dictionary(path=args.dictionary)
    n_token = len(dictionary)
    # Load Data
    print('Begin to load data.')
    traindev_data = pd.read_csv(args.traindev_data)
    if args.valid_data != "":
        data_validation = pd.read_csv(args.valid_data)
    else:
        data_validation = None
    if args.test_data != "":
        data_test = pd.read_csv(args.test_data)
    else:
        data_test = None

    data_train,data_val = split_dev(traindev_data, 0.1)
    if args.validation:
        assert data_validation is not None
        data_test = data_validation
    if args.crf:
        criterion = CRF(args.nclasses).to(device)
    else:
        criterion = nn.CrossEntropyLoss()
    model = SpanModel({'ntoken':n_token,
                 'dictionary':dictionary,
                 'ninp':args.emsize,
                 'word-vector':args.word_vector,
                 'nhid':args.nhid,
                 'nlayers':args.nlayers,
                 'dropout':args.dropout,
                 'nclasses':args.nclasses,
                 'attention':args.attention,
                 'attention-unit':args.attention_unit,
                 'num-heads':args.num_heads
                })
    model = model.to(device)
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
    elif args.optimizer == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=0.95)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
    else:
        raise Exception('For other optimizers, please add it yourself. '
                        'supported ones are: SGD, Adam and Adadelta.')             
    best_f1 = None
    best_model = None
    trainer = Trainer(data_train, dictionary, device, args, criterion=criterion, optimizer=optimizer)
    running_pat = args.patience 
    epoch = 0
    while running_pat > 0:
        trainer.data_shuffle()
        train_loss, model = trainer.epoch(epoch+1, model)
        f1,_,_ = trainer.evaluate(model, data_val, bsz=args.test_bsize)
        if best_f1 is not None and f1 < best_f1:
            running_pat -= 1
        if not best_f1 or f1 > best_f1:
            running_pat = args.patience
            best_f1 = f1
            best_model = copy.deepcopy(model)
        print('-' * 80)
        print(f'| stage 1 epoch {epoch+1} | train loss {train_loss:.8f} | dev f1 {f1:.4f} | patience left {running_pat}/{args.patience}')
        print('-' * 80)
        epoch = epoch + 1

    best_boost_f1 = best_f1
    if args.stage2 > 0:
        assert best_model is not None
        model = best_model
        model = model.to(device)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr*0.25, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
        elif args.optimizer == 'Adadelta':
            optimizer = optim.Adadelta(model.parameters(), lr=args.lr*0.25, rho=0.95)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
        trainer.update_opt(optimizer)
        running_pat = args.stage2
        epoch = 0
        while running_pat > 0:
            trainer.data_shuffle()
            train_loss, model = trainer.epoch(epoch+1, model)
            f1,_,_ = trainer.evaluate(model, data_val, bsz=args.test_bsize)
            if best_boost_f1 is not None and f1 < best_boost_f1:
                running_pat -= 1
            if not best_boost_f1 or f1 > best_boost_f1:
                running_pat = args.stage2
                best_boost_f1 = f1
                best_model = copy.deepcopy(model)
            print('-' * 75)
            print(f'| stage 2 epoch {epoch+1} | train loss {train_loss:.8f} | dev f1 {f1:.4f} | patience left {running_pat}/{args.stage2}')
            print('-' * 75)
            epoch = epoch + 1

    print(f'| best dev f1. {best_boost_f1:.4f} |')
    if data_test is not None:
        best_model = best_model.to(device)
        f1, y_pred, y_true = trainer.evaluate(best_model, data_test, bsz=args.test_bsize)
        print('-' * 75)
        print(f'| test F1 {f1:.4f} |')
        print('-' * 75)

    if args.validation:
        with open(args.log,'a') as f:
            nl = '\n'
            tab = '\t'
            tm = 'f1'
            f.write(f'{nl}{nl}{args}{nl}{tab}{tm}{tab}{f1}')
    exit(0)

