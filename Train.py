#!/usr/bin/env python
# coding: utf-8

# In[4]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import argparse
import json
import random
import time

import tqdm
import pickle
from torchvision import datasets, transforms, models
import loader
import models
import numpy as np

import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import unicodedata
import re
import datetime as dt
print(torch.__version__)
import shutil
#import convlstm


# In[ ]:


print(str(dt.datetime.now())+" Process Started")


# In[ ]:


# Set random seeds
random.seed(7)
torch.manual_seed(7)


# In[ ]:


def findcp(val):
    if val == 'e':
        ecp = []
        for i in os.listdir('/scratch/skp454/AST/MainDir/models/'):
            if 'encoder.pth' in i:
                ecp.append(int(i.split('_')[0]))
        return str(max(ecp)+1)
    if val == 'd':
        dcp = []
        for i in os.listdir('/scratch/skp454/AST/MainDir/models/'):
            if 'decoder.pth' in i:
                dcp.append(int(i.split('_')[0]))
        return str(max(dcp)+1)


# In[ ]:


def print_output(out_path, output_list):
    """
    Function that creates an output file with all of the output lines in
    the output list.
    Args:
        out_path: path to write the output file.
        output_list: list containing all the output items 
    """
    output_file = open(out_path,"a+")
    output_file.write("\n\n")
    output_file.write("\n".join(output_list))
    output_file.close()


# In[ ]:


def save_checkpoint(state, is_best, mtype, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, mtype+'_best.pth.tar')


# In[ ]:


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#device = "cpu"


# In[ ]:


with (open('/scratch/skp454/AST/MainDir/data/'+'dictionary.pkl', "rb")) as openfile:
    dictionary = pickle.load(openfile, encoding = 'latin1')


# In[5]:


dev_file = '/scratch/skp454/AST/MainDir/data/test_set.pkl'
train_file = '/scratch/skp454/AST/MainDir/data/train_set.pkl'
#train_file = '/scratch/skp454/AST/MainDir/data/test_set.pkl'


# In[2]:


BATCH_SIZE = 16
units = 256
embedding_dim = 256


# In[ ]:


print(str(dt.datetime.now())+" Loading Data")


# In[6]:


# Loaders
train_ldr = loader.make_loader(train_file, BATCH_SIZE)
dev_ldr = loader.make_loader(dev_file, BATCH_SIZE)


# In[ ]:


print(str(dt.datetime.now())+" Data Loading Complete")


# In[ ]:


def trans_input(inputs):
    inputs = np.array(inputs)
    inputs = np.swapaxes(inputs,1,3)
    inputs = np.swapaxes(inputs,2,3)
    return torch.Tensor(inputs)

def trans_label(labels):
    labels = np.array(labels)
    #labels = np.swapaxes(labels,1,2)
    return torch.Tensor(labels)


# In[ ]:


class Encoder(nn.Module):
    def __init__(self, batch_sz, enc_units):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=(1,2), padding=0),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=(1,2), padding=0),
            nn.BatchNorm2d(64))
        self.recurrent = nn.Sequential(
            nn.LSTM(4864, self.enc_units, 3, batch_first=True,  bidirectional =False, dropout = 0.5))
        
    def forward(self, x, device):
        x = self.feature(x) 
        self.hidden = self.initialize_hidden_state(device)
        x = x.permute(0,3,1,2)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        output, self.hidden = self.recurrent(x)
        return output, self.hidden

    def initialize_hidden_state(self, device):
        return torch.zeros((1, self.batch_sz, self.enc_units)).to(device)


# In[ ]:


# encoder = Encoder(BATCH_SIZE, units)
# encoder.to(device)
# enc_output, [enc_hidden, enc_cell] = encoder(inputs.to(device), device)
# print(enc_output.size())
# print(enc_hidden.size())# max_length, batch_size, enc_units


# In[ ]:


class Decoder(nn.Module):
    def __init__(self, vocab_size, dec_units, enc_units, batch_sz, embedding_dim):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.enc_units = enc_units
        self.recurrent = nn.Sequential(
            nn.LSTM(512, self.enc_units, 3, batch_first=True,  bidirectional =False, dropout = 0.5))
        self.fc = nn.Linear(self.enc_units, self.vocab_size)
        
        # used for attention
        self.W1 = nn.Linear(self.enc_units, self.dec_units)
        self.W2 = nn.Linear(self.enc_units, self.dec_units)
        self.V = nn.Linear(self.enc_units, 1)
    
    def forward(self, x, hidden, enc_output):
        
        hidden_with_time_axis = hidden.permute(1, 0, 2)[:,2,:].unsqueeze(1)
        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
        
        #score = torch.tanh(self.W2(hidden_with_time_axis) + self.W1(enc_output))
        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = torch.softmax(self.V(score), dim=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)
        
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = x.type(torch.cuda.LongTensor)
        x = x.unsqueeze(1)
        x = x.to(device)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        #x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # ? Looks like attention vector in diagram of source
        x = torch.cat((context_vector.unsqueeze(1), x), -1)
        
        # passing the concatenated vector to the GRU
        # output: (batch_size, 1, hidden_size)
        output, [state, cell] = self.recurrent(x)
        
        
        # output shape == (batch_size * 1, hidden_size)
        output =  output.view(-1, output.size(2))
        
        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)
        #x = torch.softmax(x, dim=1)
        return x, state, attention_weights
    
    def initialize_hidden_state(self):
        return torch.zeros((1, self.batch_sz, self.dec_units))


# In[ ]:


print(str(dt.datetime.now())+" Training Begins")


# In[ ]:


# decoder = Decoder(len(dictionary.keys()), units, units, BATCH_SIZE, embedding_dim)
# decoder = decoder.to(device)

# #print(enc_hidden.squeeze(0).shape)
# dec_hidden = enc_hidden#.squeeze(0)
# dec_input = labels[:,0]
# print("Decoder Input: ", dec_input.shape)
# print("--------")

# for t in range(1, labels.size(1)):
#     # enc_hidden: 1, batch_size, enc_units
#     # output: max_length, batch_size, enc_units
#     predictions, dec_hidden, _ = decoder(dec_input.to(device), 
#                                          dec_hidden.to(device), 
#                                          enc_output.to(device))


#     print("Prediction: ", predictions.shape)
#     print("Decoder Hidden: ", dec_hidden.shape)
    
#     #loss += loss_function(y[:, t].to(device), predictions.to(device))
    
#     dec_input = labels[:,t]
#     print(dec_input.shape)
#     break


# In[ ]:


criterion = nn.CrossEntropyLoss()

def loss_function(real, pred):
    """ Only consider non-zero inputs in the loss; mask needed """
    #mask = 1 - np.equal(real, 0) # assign 0 to all above 0 and 1 to all 0s
    #print(mask)
    mask = real.ge(1).type(torch.cuda.FloatTensor)
    real = real.type(torch.cuda.LongTensor)
#     mask = real.ge(1).type(torch.FloatTensor)
#     real = real.type(torch.LongTensor)
    loss_ = criterion(pred, real) * mask 
    
    return torch.mean(loss_)
    


# In[ ]:


## TODO: Combine the encoder and decoder into one class
encoder = Encoder(BATCH_SIZE, units)
decoder = Decoder(len(dictionary.keys())+1, units, units, BATCH_SIZE, embedding_dim)

#encoder = nn.DataParallel(encoder)
#decoder = nn.DataParallel(decoder)

encoder.to(device)
decoder.to(device)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), 
                       lr=0.001)


# In[ ]:


EPOCHS = 100

tloss = 10 

for epoch in range(EPOCHS):
    
    #### Train on train set
    start = time.time()
    
    encoder.train()
    decoder.train()
    
    total_loss = 0
    batch = 1
    for inputs, labels in train_ldr:
        inputs, labels = trans_input(inputs).to(device), trans_label(labels).to(device)
        loss = 0
        enc_output, [enc_hidden, enc_cell] = encoder(inputs.to(device), device)
        del inputs
        del enc_cell
        dec_hidden = enc_hidden
        del enc_hidden
        dec_input = labels[:,0]
        
        for t in range(1, labels.size(1)):
            predictions, dec_hidden, _ = decoder(dec_input.to(device), 
                                                 dec_hidden.to(device), 
                                                 enc_output.to(device))
            del dec_input
            loss += loss_function(labels[:,t], predictions.to(device))
            dec_input = labels[:,t]
            
        batch_loss = (loss / int(labels.size(1)))
        total_loss += batch_loss
        
        optimizer.zero_grad()
        
        loss.backward()

        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        if batch % 5 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.detach().item()))
        batch += 1
       
    del labels
    del dec_input
    del dec_hidden
    
    torch.cuda.empty_cache()
    
    ### Evaluation on dev_set
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        dev_total_loss = 0
        dev_batch = 0
        for inputs, labels in dev_ldr:
            inputs, labels = trans_input(inputs).to(device), trans_label(labels).to(device)
            loss = 0
            enc_output, [enc_hidden, enc_cell] = encoder(inputs.to(device), device)
            del inputs
            del enc_cell
            dec_hidden = enc_hidden
            del enc_hidden
            dec_input = labels[:,0]

            for t in range(1, labels.size(1)):
                predictions, dec_hidden, _ = decoder(dec_input.to(device), 
                                                     dec_hidden.to(device), 
                                                     enc_output.to(device))
                del dec_input
                loss += loss_function(labels[:,t], predictions.to(device))
                dec_input = labels[:,t]

            dev_batch_loss = (loss / int(labels.size(1)))
            dev_total_loss += dev_batch_loss
            if dev_batch % 5 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         dev_batch,
                                                         dev_batch_loss.detach().item()))
            dev_batch+=1
            

        lis=[]
        lis.append('Time{}:'.format(dt.datetime.now()))
        lis.append('Epoch {} Train Loss {:.4f} Dev Loss {:.4f}'.format(epoch + 1,
                   total_loss / len(train_ldr), dev_total_loss / len(dev_ldr)))
        lis.append('Time taken for 1 epoch {} sec'.format(time.time() - start))


        print_output('logs.txt', lis)
        print(lis)

        is_best = tloss > (dev_total_loss / len(dev_ldr))

        if is_best:
            tloss = (dev_total_loss / len(dev_ldr))

    if epoch % 1 == 0:
        # save encoder
        save_checkpoint({
                             'epoch': epoch + 1,
                             'arch': 'encoder',
                             'state_dict': encoder.state_dict(),
                             'best_acc': tloss,
                             'optimizer' : optimizer.state_dict(),
                         }, is_best, mtype ='encoder', filename= \
            '/scratch/skp454/AST/MainDir/models/'+findcp('e')+'_encoder.pth.tar')

        # save decoder
        save_checkpoint({
                             'epoch': epoch + 1,
                             'arch': 'decoder',
                             'state_dict': decoder.state_dict(),
                             'best_acc': tloss,
                             'optimizer' : optimizer.state_dict(),
                         }, is_best, mtype ='decoder', filename= \
            '/scratch/skp454/AST/MainDir/models/'+findcp('d')+'_decoder.pth.tar')

