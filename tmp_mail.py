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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

##################Usage python main.py <usage_mode:eval, train>
    try:
        if(sys.argv[2]=='eval'):
            model = torch.load(sys.argv[3])
            print('Model ', sys.argv[3], ' loaded successfully')
            model.eval()
            Question = input("Insert a Question for the model: ")
            #Handle the passing of the user inserted input to the model
    except IndexError:
        print('No model to load')  

    #define the pad index
    src_pad_idx = 0
    trg_pad_idx = 0
    #Define the vocab size
    src_vocab_size = 73
    trg_vocab_size = 73
    
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
    model.to(device)
    start_time = time.time()
    temp = start_time
    total_loss = 0
    epochs = 10
    #define criterion:
    criterion = nn.CrossEntropyLoss(ignore_index=0) #ignore padding index
    steps = 0
    optim.zero_grad()
    loss_sum, acc_sum, loss_counter = 0, 0, 0
    accum_loss, accum_acc = 0, 0

    for epoch in range(epochs):
        print('EPOCH: ', epoch)
        for file in filelist:
            train_iter=get_train_iterator(file, int(sys.argv[1]))
            for i, batch in enumerate(train_iter): #l'enumerate finisce non va avanti all'infinito
                src = batch[0]
                trg = batch[1]
                src = src.to(device)
                trg = trg.to(device)
                #print(batch[0].shape)
                #print(batch[1].shape)
                #print(i)       
                steps+=1
                logits = model(src, trg[:, :-1])

            # [batch_size, trg_seq_len-1, output_dim]

                flat_logits = logits.contiguous().view(-1, logits.shape[-1])
                # [batch_size * (trg_seq_len-1), output_dim]

                # ignore SOS symbol (skip first)
                flat_trg = trg[:, 1:].contiguous().view(-1)
                # [batch_size * (trg_seq_len-1)]

                # compute loss
                loss = criterion(flat_logits, flat_trg) #no division by acc steps because = 1
                        #With loss.backward we compute all the gradients
                #print('LOSS: ', loss)
                loss.backward()
                ### Now we need an optimizer to do the training step
                # compute acc
                acc = compute_accuracy(logits=logits,
                                        targets=trg,
                                        pad_value=0)
                accum_loss += loss
                accum_acc += acc
                #grad_accum_step += 1
                #if grad_accum_step % p.grad_accum_steps == 0
                #we perform step each time
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                       max_norm=0.1) #default 0.1
                optim.step()
                optim.zero_grad()
                #grad_accum_step = 0
                # update loss and acc sum and counter
                loss_counter += 1
                loss_sum += accum_loss
                acc_sum += accum_acc

                # reset accum values
                accum_loss = 0
                accum_acc = 0

                if steps % 50 == 0 and steps != 0:
                # compute loggin metrics
                    elapsed = time.time() - start_time
                    steps_per_sec = loss_counter / elapsed
                    start_time = time.time()
                    avg_loss = loss_sum / loss_counter
                    avg_acc = acc_sum / loss_counter
                    loss_sum, acc_sum, loss_counter = 0, 0, 0
                    print('STEP: ', steps, ' TIME ELAPSED ',elapsed, ' avg loss :', avg_loss, ' avg accuracy: ', avg_acc )
                            
    torch.save(model, 'model.pt')
    print('Model succesfully saved')

print('You are in the main file')