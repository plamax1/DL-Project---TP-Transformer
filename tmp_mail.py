from tp_transformer import *
#from test_dataset_loading import *
from dataset_loading import *
import torch
import time
import torch.nn as nn
from utils import *
import sys
import glob
import os
path = pathlib.Path().resolve()
target_path = os.path.join(path, 'Dataset/**/*.txt')
#import test_dataset_loading
'''
def get_readable_output (input, train_iter):
    for i in input:
        # in i there is the single sentence
        string = ''
        for j in list(i):
            string.append(train_iter.int(j))
'''
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    #define the pad index
    src_pad_idx = 0
    trg_pad_idx = 0
    #Define the vocab size
    src_vocab_size = 130
    trg_vocab_size = 130
    
    #Create model
    print('Creating model...')
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
        device
    )
    filelist=[]
    ### Dataset loading
    ###We choose to load one file at once and train the model on that file to 
    #avoid going out of ram
    for file in glob.iglob(target_path, recursive=True):
        filelist.append(file)
     #   train_iter= get_train_iterator(file , int(sys.argv[1]))
    #train_iter = get_demo_trainer(int(sys.argv[1]))
    print('Model created')
    #Now how to train the model?
    print('Starting model training')
    #Defining optimizer
    optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    model.train()
    start = time.time()
    temp = start
    total_loss = 0
    epochs = 1
    #define criterion:
    criterion = nn.CrossEntropyLoss(ignore_index=0) #ignore padding index

    for epoch in range(epochs):
        for file in filelist:
            train_iter=get_train_iterator(file, int(sys.argv[1]))
            for i, batch in enumerate(train_iter): #l'enumerate finisce non va avanti all'infinito
                src = batch[0]
                trg = batch[1]
                src = src.to(device)
                trg = trg.to(device)
                #print(batch[0].shape)
                #print(batch[1].shape)
                print(i)       

                logits = model(src, trg[:, :-1])
                optim.zero_grad()

            # [batch_size, trg_seq_len-1, output_dim]

                flat_logits = logits.contiguous().view(-1, logits.shape[-1])
                # [batch_size * (trg_seq_len-1), output_dim]

                # ignore SOS symbol (skip first)
                flat_trg = trg[:, 1:].contiguous().view(-1)
                # [batch_size * (trg_seq_len-1)]

                # compute loss
                loss = criterion(flat_logits, flat_trg)
                        #With loss.backward we compute all the gradients
                print('LOSS: ', loss)
                loss.backward()
                ### Now we need an optimizer to do the training step
                # compute acc
                optim.step()
                total_loss += int(loss)
                if (i + 1) % 10 == 0:
                    loss_avg = total_loss / 100
                    print('------------------------------------------')
                    print('AVG LOSS UP TO NOW**************************: ', loss_avg)
                    total_loss = 0
                    acc = compute_accuracy(logits=logits,
                                        targets=trg,
                                        pad_value=0)
                    print('ACCURACY: ', acc)
                


print('You are in the main file')