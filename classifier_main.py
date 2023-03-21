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
from tp_transformer_pl import *

import os
import pathlib
path = pathlib.Path().resolve()
target_path = os.path.join(path, 'Dataset/**/*.txt')
#import test_dataset_loading
import pytorch_lightning as pl
import torch 
   # Use NLLLoss()

def add_padding(data, max_lenght):
    result=[]
    j=0
    #print('add padding input shape: ', data.shape)
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
    try:
        model_name = sys.argv[1]
        batch_size = sys.argv[2]
    except:
        print('Usage python main.py model_name batch_size')
        exit(0)


    # Creating vocab
    voc = Vocabulary(73) #73 is the vocabulary len used in the paper
    #build vocab
    voc.build_vocabulary()
    print('VOCABULARY CREATED')
    #Create model
    print('Creating model...')
    print('ARGV[1]=', sys.argv[1])
    
    if(sys.argv[1].strip()=='ask'):
        model = torch.load('tp-transformer.pt')
        print('Model loaded successfully')
        while(1):
                question = input("Insert a Question for the model: ")
                tkq=torch.tensor([voc.stoi['<SOS>']])
                tkq= torch.cat((tkq, torch.tensor(voc.numericalize(list(question))), torch.tensor([voc.stoi['<EOS>']])))
                print('passing to the model: ', tkq)
                print('Stringa prodotta: ', tensor_to_string(voc, tkq))
                preds = model(tkq)
                print(tensor_to_string(preds[1:-1]))

    if(sys.argv[1].strip()=='tp-transformer'):
        model = Transformer(73, 73, 0, 0, device='cpu').to(device)
    if(sys.argv[1].strip()=='classifier'):
        model = Multiclass(200, 35, 73)
        print('Model built')
    if(not(sys.argv[1]=='classifier' or sys.argv[1]=='tp-transformer')):
        print('Model not implemented yet, you can choose tp-transformer, classifier')
        exit(0)

    #Pre-loading dataset files:
    filelist=[]
    for file in glob.iglob(target_path, recursive=True):
        filelist.append(file)
        #print('Trainer file list loaded')
    model.to(device)

    trainer = pl.Trainer()
    train_iterator = get_train_iterator('test.txt', 3, voc)
    print('train_it', type(train_iterator))
    #trainer.fit(model, train_iterator)
    trainer.fit(model, train_dataloaders = train_iterator)