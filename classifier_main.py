#from test_dataset_loading import *
from dataset_loading import Vocabulary, get_train_iterator, tensor_to_string, get_test_iterator
import torch
import time
import torch.nn as nn
from utils import *
from classifier_model import Multiclass
import sys
import glob
import torch.optim as optim

import os
import pathlib
path = pathlib.Path().resolve()
target_path = os.path.join(path, 'Dataset/**/*.txt')
#import test_dataset_loading

import torch 
   # Use NLLLoss()

def add_padding(data, max_lenght):
    result=[]
    j=0
    print('add padding input shape: ', data.shape)
    for i in data:
        ln = len(i)
        #print('ln: ', ln)
        to_add = max_lenght-ln
        result.append((torch.cat((i, torch.zeros(to_add)))))
        #print('ishape: ', i.shape)
        j+=1
    result= torch.stack(result)
    #print('result shape: ', result.shape)
    return result

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Creating vocab
    voc = Vocabulary(73) #73 is the vocabulary len used in the paper
    #build vocab
    voc.build_vocabulary()
    print('VOCABULARY CREATED')
    print('Vocabulary lenght: ', len(voc))
    #print('index to string: ',voc.itos)
    #print('string to index:',voc.stoi)
    #Create model
    print('Creating model...')
    model = Multiclass(200, 35, 73)
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #Pre-loading dataset files:
    filelist=[]
    for file in glob.iglob(target_path, recursive=True):
        filelist.append(file)
        #print('Trainer file list loaded')
    model.to(device)
    model.train()

    epochs = 2
    steps=0
    for epoch in range(epochs):
        print('Epoch: ', epoch)
        running_loss = 0.0
        for file in filelist:
            print('Loading file : ', file)
            train_iter=get_train_iterator(file, 10, voc)
            for i, batch in enumerate(train_iter):
                inputs = add_padding(batch[0], 200)
                labels = batch[1]
                # set optimizer to zero grad to remove previous epoch gradients
                optimizer.zero_grad()
                # forward propagation
                outputs = model(inputs.float())
                print('Output shape: ', outputs.shape)
                loss = loss_fn(outputs, labels)
                # backward propagation
                loss.backward()
                # optimize
                optimizer.step()
                running_loss += loss.item()
                steps+=1
                if(steps%100==0):
                    # display statistics
                    print('STEP: ',steps , ' ', f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
