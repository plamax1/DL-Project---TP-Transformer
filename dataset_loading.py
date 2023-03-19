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
import pathlib
import os
'''
from other_test_dataset import *
a=get_test()
for i, batch in enumerate(a): #l'enumerate finisce non va avanti all'infinito
                src = batch[0]
                trg = batch[1]
                print(batch[0].shape)
                print(batch[1].shape)
                print(i)'''

class Vocabulary:
  
    '''
    __init__ method is called by default as soon as an object of this class is initiated
    we use this method to initiate our vocab dictionaries
    '''
    def __init__(self, max_size):
        '''
        max_size : max source vocab size. Eg. if set to 10,000, we pick the top 10,000 most frequent words and discard others
        '''
        #initiate the index to token dict
        ## <PAD> -> padding, used for padding the shorter sentences in a batch to match the length of longest sentence in the batch
        ## <SOS> -> start token, added in front of each sentence to signify the start of sentence
        ## <EOS> -> End of sentence token, added to the end of each sentence to signify the end of sentence
        ## <unk> -> Unknown token
        self.itos = {0: '<PAD>',1: '<UNK>', 2:'<SOS>', 3:'<EOS>'}
        #initiate the token to index dict
        self.stoi = {k:j for j,k in self.itos.items()} 
        
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
        #return [tok.lower() for tok in split_chars(text)]
        return split_chars(text)
    
    '''
    build the vocab: create a dictionary mapping of index to string (itos) and string to index (stoi)
    output ex. for stoi -> {'the':5, 'a':6, 'an':7}
    '''
    def build_vocabulary(self):
        idx = 4 #index from which we want our dict to start. We already used 4 indexes for pad, start, end, unk
        #init the vocab
        words = self.tokenizer(" e*t-i12osa.3r()hn40+5dlp6mcf=u78/9vLb,?gWwSyqkxzjC:IDFEPMGR{}'AHT<>!")
            
        #create vocab
        for word in words:
            print('Associating  token: ', word , 'with index: ', idx)
            self.stoi[word] = idx
            self.itos[idx] = word
            idx+=1
            
    '''
    convert the list of token to a list of corresponding indexes
    '''    
    def numericalize(self, text):
        #tokenize text
        tokenized_text = self.tokenizer(text)
        numericalized_text = []
        for token in tokenized_text:
            if token in self.stoi.keys():
                numericalized_text.append(self.stoi[token])
            else: #out-of-vocab words are represented by UNK token index
                numericalized_text.append(self.stoi['<UNK>'])
                
        return numericalized_text



#create a vocab class with freq_threshold=0 and max_size=100
voc = Vocabulary(73)
#build vocab
voc.build_vocabulary()
print('VOCABULARY CREATED')
print('Vocabulary lenght: ', len(voc))

print('index to string: ',voc.itos)
print('string to index:',voc.stoi)

print('Starting Dataset pre-processing')

#######################################################
#               Define Vocabulary Class
#######################################################





#print('numericalize -> cat and a dog: ', voc.numericalize('cat and a dog'))


class Train_Dataset(Dataset):
    '''
    Initiating Variables
    df: the training dataframe
    source_column : the name of source text column in the dataframe
    target_columns : the name of target text column in the dataframe
    transform : If we want to add any augmentation
    freq_threshold : the minimum times a word must occur in corpus to be treated in vocab
    source_vocab_max_size : max source vocab size
    target_vocab_max_size : max target vocab size
    '''
    
    def __init__(self, df, source_column, target_column, transform=None,
                source_vocab_max_size = 200, target_vocab_max_size = 200):
    
        self.df = df
        self.transform = transform
        
        #get source and target texts
        self.source_texts = self.df[source_column]
        self.target_texts = self.df[target_column]
        
        
        ##VOCAB class has been created above
        #Initialize source vocab object and build vocabulary
        self.source_vocab = Vocabulary(source_vocab_max_size)
        self.source_vocab.build_vocabulary()
        #Initialize target vocab object and build vocabulary
        self.target_vocab = Vocabulary(target_vocab_max_size)
        self.target_vocab.build_vocabulary()
        
    def __len__(self):
        return len(self.df)
    
    '''
    __getitem__ runs on 1 example at a time. Here, we get an example at index and return its numericalize source and
    target values using the vocabulary objects we created in __init__
    '''
    def __getitem__(self, index):
        source_text = self.source_texts[index]
        target_text = self.target_texts[index]
        
        if self.transform is not None:
            source_text = self.transform(source_text)
            
        numerialized_source = [self.source_vocab.stoi["<SOS>"]]
        numerialized_source += self.source_vocab.numericalize(source_text)
        numerialized_source.append(self.source_vocab.stoi["<EOS>"])
    
        numerialized_target = [self.target_vocab.stoi["<SOS>"]]
        numerialized_target += self.target_vocab.numericalize(target_text)
        numerialized_target.append(self.target_vocab.stoi["<EOS>"])
        
        #convert the list to tensor and return
        return torch.tensor(numerialized_source), torch.tensor(numerialized_target) 
#######################################################
#               Define Dataset Class
#######################################################

class Validation_Dataset:
    def __init__(self, train_dataset, df, source_column, target_column, transform = None):
        self.df = df
        self.transform = transform
        
        #train dataset will be used as lookup for vocab
        self.train_dataset = train_dataset
        
        #get source and target texts
        self.source_texts = self.df[source_column]
        self.target_texts = self.df[target_column]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        source_text = self.source_texts[index]
        #print(source_text)
        target_text = self.target_texts[index]
        #print(target_text)
        if self.transform is not None:
            source_text = self.transform(source_text)
            
        numerialized_source = [self.train_dataset.source_vocab.stoi["<SOS>"]]
        numerialized_source += self.train_dataset.source_vocab.numericalize(source_text)
        numerialized_source.append(self.train_dataset.source_vocab.stoi["<EOS>"])
    
        numerialized_target = [self.train_dataset.target_vocab.stoi["<SOS>"]]
        numerialized_target += self.train_dataset.target_vocab.numericalize(target_text)
        numerialized_target.append(self.train_dataset.target_vocab.stoi["<EOS>"])
        return torch.tensor(numerialized_source), torch.tensor(numerialized_target)
    
#######################################################
#               Collate fn 
#######################################################

'''
class to add padding to the batches
collat_fn in dataloader is used for post processing on a single batch. Like __getitem__ in dataset class
is used on single example
'''

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        #Takes in input the integer used for the PAD
    
    #__call__: a default method
    ##   First the obj is created using MyCollate(pad_idx) in data loader
    ##   Then if obj(batch) is called -> __call__ runs by default
    def __call__(self, batch):
        #get all source indexed sentences of the batch
        source = [item[0] for item in batch] 
        #pad them using pad_sequence method from pytorch. 
        source = pad_sequence(source, batch_first=False, padding_value = self.pad_idx) 
        
        #get all target indexed sentences of the batch
        target = [item[1] for item in batch] 
        #pad them using pad_sequence method from pytorch. 
        target = pad_sequence(target, batch_first=False, padding_value = self.pad_idx)
        #return source, target
        return torch.transpose(source, 0,1), torch.transpose(target, 0,1)

#######################################################
#            Define Dataloader Functions
#######################################################

# If we run a next(iter(data_loader)) we get an output of batch_size * (num_workers+1)
def get_train_loader(dataset, batch_size, num_workers=0, shuffle=True, pin_memory=True): #increase num_workers according to CPU
    #get pad_idx for collate fn
    pad_idx = dataset.source_vocab.stoi['<PAD>']
    #define loader
    loader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers,
                        shuffle=shuffle,
                       pin_memory=pin_memory, collate_fn = MyCollate(pad_idx=pad_idx)) #MyCollate class runs __call__ method by default
    return loader

def get_valid_loader(dataset, train_dataset, batch_size, num_workers=0, shuffle=True, pin_memory=True):
    pad_idx = train_dataset.source_vocab.stoi['<PAD>']
    loader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers,
                        shuffle=shuffle,
                       pin_memory=pin_memory, collate_fn = MyCollate(pad_idx=pad_idx))
    return loader


#train_dataset = Train_Dataset(data, 'Question', 'Answer')
#print(train.loc[1])
#train_dataset[1]

#train_loader = get_train_loader(train_dataset, 32)
#def get_demo_trainer(batch_size):
 #   return get_train_loader(train_dataset, batch_size)

#source = next(iter(train_loader))[0]
#target = next(iter(train_loader))[1]
###Quindi il dataloader restituisce batch size, tutti della stessa lunghezza. Es se la max len di una sentence in un batch_size è 120
#allora tutte le sentence avranno lunghezza 120, chi sta a meno si aggiunge il padding.

def get_train_iterator(filename, batch_size):
    data=[]
    with open(filename) as f:
        lines = f.readlines()
        print('Loading file : ', filename, ' file-len: ',  len(lines))

    stripped=[]
    for i in lines:
        stripped.append(i.strip())

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
    print("DataFrame Created")
    train_dataset = Train_Dataset(data, 'Question', 'Answer')
    return get_train_loader(train_dataset, batch_size)

