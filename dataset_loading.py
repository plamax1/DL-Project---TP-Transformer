# sys libs
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from string import digits
from pandarallel import pandarallel
import pandas as pd
import pathlib
import glob
import os
import warnings
warnings.filterwarnings("ignore")

path = pathlib.Path().resolve()
test_path = os.path.join(path, 'TestDataset/**/*.txt')

# data manupulation libs
# Initialization
pandarallel.initialize()


# string manupulation libs

# torch libs

# Source of Inspiration URL:
# https://towardsdatascience.com/custom-datasets-in-pytorch-part-2-text-machine-translation-71c41a3e994e

# Define Vocabulary Class


class Vocabulary:

    '''
    __init__ method is called by default as soon 
    as an object of this class is initiated
    we use this method to initiate our vocab dictionaries
    '''

    def __init__(self, max_size):

        # initiate the index to token dict
        # <PAD> -> padding, used for padding the shorter
        # sentences in a batch to match the
        # length of longest sentence in the batch
        # <SOS> -> start token, added in front of
        # each sentence to signify the start of sentence
        # <EOS> -> End of sentence token, added
        # to the end of each sentence to signify the end of sentence
        # <unk> -> Unknown token
        self.itos = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        # initiate the token to index dict
        self.stoi = {k: j for j, k in self.itos.items()}

        self.max_size = max_size

    '''
    __len__ is used by dataloader later to create batches
    '''

    def __len__(self):
        return len(self.itos)

    '''
    a simple tokenizer to split on space and 
    converts the sentence to list of words
    '''
    @staticmethod
    def tokenizer(text):
        def split_chars(x): return list(x)
        return split_chars(text)

    '''
    build the vocab: create a dictionary mapping 
    of index to string (itos) and string to index (stoi)
    output ex. for stoi -> {'the':5, 'a':6, 'an':7}
    '''

    def build_vocabulary(self):
        idx = 4
        # index from which we want our dict to start.
        # We already used 4 indexes for pad, start, end, unk
        # init the vocab
        words = self.tokenizer(
            " e*t-i12osa.3r()hn40+5dlp6mcf=u78/9vLb,?gWwSyqkxzjC:IDFEPMGR{}'AHT<>!")

        # create vocab
        for word in words:
            # print('Associating  token: ', word , 'with index: ', idx)
            self.stoi[word] = idx
            self.itos[idx] = word
            idx += 1

    '''
    convert the list of token to a list of corresponding indexes
    '''

    def numericalize(self, text):
        # tokenize text
        tokenized_text = self.tokenizer(text)
        numericalized_text = []
        for token in tokenized_text:
            if token in self.stoi.keys():
                numericalized_text.append(self.stoi[token])
            else:  # out-of-vocab words are represented by UNK token index
                numericalized_text.append(self.stoi['<UNK>'])

        return numericalized_text


print('Starting Dataset pre-processing')


class Train_Dataset(Dataset):
    def __init__(self, df, source_column,
                 target_column, vocab, transform=None):

        self.df = df
        self.transform = transform

        # get source and target texts
        self.source_texts = self.df[source_column]
        self.target_texts = self.df[target_column]

        # VOCAB class has been created above
        # Initialize source vocab object and build vocabulary
        self.source_vocab = vocab
        # Initialize target vocab object and build vocabulary
        self.target_vocab = vocab

    def __len__(self):
        return len(self.df)

    '''
    __getitem__ runs on 1 example at a time. Here, we get 
    an example at index and return its numericalize source and
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

        # convert the list to tensor and return
        return torch.tensor(numerialized_source), torch.tensor(
            numerialized_target)
#######################################################
#               Define Dataset Class
#######################################################


class Validation_Dataset:
    def __init__(self, train_dataset, df,
                 source_column, target_column, transform=None):
        self.df = df
        self.transform = transform

        # train dataset will be used as lookup for vocab
        self.train_dataset = train_dataset

        # get source and target texts
        self.source_texts = self.df[source_column]
        self.target_texts = self.df[target_column]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        source_text = self.source_texts[index]
        # print(source_text)
        target_text = self.target_texts[index]
        # print(target_text)
        if self.transform is not None:
            source_text = self.transform(source_text)

        numerialized_source = [self.train_dataset.source_vocab.stoi["<SOS>"]]
        numerialized_source += self.train_dataset.source_vocab.numericalize(
            source_text)
        numerialized_source.append(
            self.train_dataset.source_vocab.stoi["<EOS>"])

        numerialized_target = [self.train_dataset.target_vocab.stoi["<SOS>"]]
        numerialized_target += self.train_dataset.target_vocab.numericalize(
            target_text)
        numerialized_target.append(
            self.train_dataset.target_vocab.stoi["<EOS>"])
        return torch.tensor(numerialized_source
                            ), torch.tensor(numerialized_target)


# Collate fn


'''
class to add padding to the batches
collat_fn in dataloader is used for
post processing on a single batch. Like __getitem__ in dataset class
is used on single example
'''


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        # Takes in input the integer used for the PAD

    # __call__: a default method
    # First the obj is created using MyCollate(pad_idx) in data loader
    # Then if obj(batch) is called -> __call__ runs by default
    def __call__(self, batch):
        # get all source indexed sentences of the batch
        source = [item[0] for item in batch]
        # pad them using pad_sequence method from pytorch.
        source = pad_sequence(source, batch_first=False,
                              padding_value=self.pad_idx)

        # get all target indexed sentences of the batch
        target = [item[1] for item in batch]
        # pad them using pad_sequence method from pytorch.
        target = pad_sequence(target, batch_first=False,
                              padding_value=self.pad_idx)
        # return source, target
        return torch.transpose(source, 0, 1), torch.transpose(target, 0, 1)

# Define Dataloader Functions


# If we run a next(iter(data_loader))
# we get an output of batch_size * (num_workers+1)
# increase num_workers according to CPU
def get_train_loader(dataset, batch_size,
                     num_workers=0, shuffle=True, pin_memory=True):
    # get pad_idx for collate fn
    pad_idx = dataset.source_vocab.stoi['<PAD>']
    # define loader
    loader = DataLoader(dataset, batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        pin_memory=pin_memory,
                        collate_fn=MyCollate(pad_idx=pad_idx))
    # MyCollate class runs __call__ method by default
    return loader


def get_valid_loader(dataset, train_dataset, batch_size,
                     num_workers=0, shuffle=True, pin_memory=True):
    pad_idx = train_dataset.source_vocab.stoi['<PAD>']
    loader = DataLoader(dataset, batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        pin_memory=pin_memory,
                        collate_fn=MyCollate(pad_idx=pad_idx))
    return loader


def get_train_iterator(train_path, batch_size, vocab, percentage):
    data = []
    for file in glob.iglob(train_path, recursive=True):

        with open(file) as f:
            lines = f.readlines()
            lines = lines[:int(len(lines)*percentage)]
            # print('Loading file : ', file, ' file-len: ',  len(lines))

        stripped = []
        for i in lines:
            stripped.append(i.strip())

        i = 0
        while i < len(stripped)-1:
            data.append([stripped[i], stripped[i+1]])
            i = i+2
    print('Dataset loaded: Loaded ', len(data),
          ' items, percentage: ', percentage)

    data = pd.DataFrame(data, columns=['Question', 'Answer'])
    train_dataset = Train_Dataset(data, 'Question', 'Answer', vocab)
    return get_train_loader(train_dataset, batch_size)


def tensor_to_string(vocab, input):
    result = ''
    for i in list(input):
        # print(i)
        result += vocab.itos[int(i)]
    return result

def get_algebra_iterator(path, batch_size, vocab, percentage):
    data = []
    for file in glob.iglob(path, recursive=True):

        with open(file) as f:
            lines = f.readlines()
            lines = lines[:int(len(lines)*percentage)]
            # print('Loading file : ', file, ' file-len: ',  len(lines))

        stripped = []
        for i in lines:
            stripped.append(i.strip())

        i = 0
        while i < len(stripped)-1:
            data.append([stripped[i], stripped[i+1]])
            i = i+2
    print('Dataset loaded: Loaded ', len(data),
          ' items, percentage: ', percentage)

    data_train = pd.DataFrame(data[:len(data)*0.9], columns=['Question', 'Answer'])
    data_test = pd.DataFrame(data[len(data)*0.9:], columns=['Question', 'Answer'])
    
    train_dataset = Train_Dataset(data_train, 'Question', 'Answer', vocab)
    test_dataset = Train_Dataset(data_test, 'Question', 'Answer', vocab)
    return get_train_loader(train_dataset, batch_size), get_train_loader(test_dataset, batch_size)
