#sys libs
import os
import sys
import random
import warnings
warnings.filterwarnings("ignore")

#data manupulation libs
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandarallel import pandarallel
# Initialization
pandarallel.initialize()


#string manupulation libs
import re
import string
from string import digits
import spacy

#torch libs
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset


with open('test.txt') as f:
    lines = f.readlines()

stripped=[]
for i in lines:
    stripped.append(i.strip())
data=[]
i=0
while i<len(stripped)-1:
    data.append([stripped[i], stripped[i+1]])
    i=i+2

data=pd.DataFrame(data,columns=['Question','Answer'])
val_frac = 0.1 #precentage data in val
val_split_idx = int(len(data)*val_frac) #index on which to split
data_idx = list(range(len(data))) #create a list of ints till len of data
np.random.shuffle(data_idx)

#get indexes for validation and train
val_idx, train_idx = data_idx[:val_split_idx], data_idx[val_split_idx:]
print('len of train: ', len(train_idx))
print('len of val: ', len(val_idx))

#create the sets
train = data.iloc[train_idx].reset_index().drop('index',axis=1)
val = data.iloc[val_idx].reset_index().drop('index',axis=1)
print("Test")

#######################################################
#               Define Vocabulary Class
#######################################################

class Vocabulary:
  
    '''
    __init__ method is called by default as soon as an object of this class is initiated
    we use this method to initiate our vocab dictionaries
    '''
    def __init__(self, freq_threshold, max_size):
        '''
        freq_threshold : the minimum times a word must occur in corpus to be treated in vocab
        max_size : max source vocab size. Eg. if set to 10,000, we pick the top 10,000 most frequent words and discard others
        '''
        #initiate the index to token dict
        ## <PAD> -> padding, used for padding the shorter sentences in a batch to match the length of longest sentence in the batch
        ## <SOS> -> start token, added in front of each sentence to signify the start of sentence
        ## <EOS> -> End of sentence token, added to the end of each sentence to signify the end of sentence
        self.itos = {0: '<PAD>', 1:'<SOS>', 2:'<EOS>'}
        #initiate the token to index dict
        self.stoi = {k:j for j,k in self.itos.items()} 
        
        self.freq_threshold = freq_threshold
        self.max_size = max_size
    
    '''
    __len__ is used by dataloader later to create batches
    '''
    def __len__(self):
        return len(self.itos)
    
    '''
    a simple tokenizer to split on space and converts the sentence to list of words
    '''
    @staticmethod
    def tokenizer(text):
        split_chars = lambda x: list(x)
        return [tok.lower() for tok in split_chars(text)]
    
    '''
    build the vocab: create a dictionary mapping of index to string (itos) and string to index (stoi)
    output ex. for stoi -> {'the':5, 'a':6, 'an':7}
    '''
    def build_vocabulary(self, sentence_list):
        #calculate the frequencies of each word first to remove the words with freq < freq_threshold
        frequencies = {}  #init the freq dict
        idx = 3 #index from which we want our dict to start. We already used 4 indexes for pad, start, end, unk
        
        #calculate freq of words
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                if word not in frequencies.keys():
                    frequencies[word]=1
                else:
                    frequencies[word]+=1
                    
                    
        #limit vocab by removing low freq words
        frequencies = {k:v for k,v in frequencies.items() if v>self.freq_threshold} 
        
        #limit vocab to the max_size specified
        frequencies = dict(sorted(frequencies.items(), key = lambda x: -x[1])[:self.max_size-idx]) # idx =4 for pad, start, end , unk
            
        #create vocab
        for word in frequencies.keys():
            self.stoi[word] = idx
            self.itos[idx] = word
            idx+=1
            
    '''
    convert the list of words to a list of corresponding indexes
    '''    
    def numericalize(self, text):
        #tokenize text
        tokenized_text = self.tokenizer(text)
        numericalized_text = []
        for token in tokenized_text:
            if token in self.stoi.keys():
                numericalized_text.append(self.stoi[token])
            #else: #out-of-vocab words are represented by UNK token index
                #numericalized_text.append(self.stoi['<UNK>'])
                
        return numericalized_text




#create a vocab class with freq_threshold=0 and max_size=100
voc = Vocabulary(0, 100)
sentence_list = ['that is a cat', 'that is not a dog']
#build vocab
voc.build_vocabulary(sentence_list)

print('index to string: ',voc.itos)
print('string to index:',voc.stoi)

print('numericalize -> cat and a dog: ', voc.numericalize('cat and a dog'))
