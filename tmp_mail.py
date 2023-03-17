from tp_transformer import *
from test_dataset_loading import *
import torch
import time
import torch.nn as nn
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

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
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

    ### Dataset loading
    train_iter = get_demo_trainer(512)
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
        for i, batch in enumerate(train_iter): #l'enumerate finisce non va avanti all'infinito
            src = batch[0]
            trg = batch[1]
            #print(batch[0].shape)
            #print(batch[1].shape)
            print(i)
            

    #in model si chama la forward di transformer
            '''preds = model(src,trg[:, :-1])
            
            print('PREDICTED: ', preds)
            print('PREDICTED SHAPE: ', preds.shape)
            print('TARGET SHAPE: ', trg.shape)
            print('Test.................')
            a=torch.flatten(preds)
            print('Flatten pred len: ', len(a))
            print('Sum of all the value in predit: ', torch.sum(a))


            #prediction shape: [batch_size, seq_lenght, trg_vocab_size]
            #now we want to apply an argmax to get the predicted token
            predicted_outputs = torch.argmax(preds, dim = 2)
            print('TRAINING LOOP: Predected Output after argmax', predicted_outputs.shape)
            print(predicted_outputs)
            #just for curiosity convert to string 
            #readable_output = get_readable_output(predicted_outputs, train_iter)
            #once we got the output what are we going to do?
           #############The model should be completed, now complete the training loop
            # compute loss
            #loss = criterion(flat_logits, flat_trg) / self.p.grad_accum_steps
            #the criterion is the loss function
            #loss = criterion(preds, trg) / self.p.grad_accum_steps
            print('LOSSPRINT: preicted_outputs shape: ', predicted_outputs.shape)
            print('LOSSPRINT: trg shape: ', trg.shape)

            flat_logits = preds.contiguous().view(-1, preds.shape[-1])
            # [batch_size * (trg_seq_len-1), output_dim]

            # ignore SOS symbol (skip first)
            flat_trg = trg[:, 1:].contiguous().view(-1)
            # [batch_size * (trg_seq_len-1)]

            loss = criterion(preds, trg)
            print('LOSS: ', loss)'''
        

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
            '''acc = compute_accuracy(logits=logits,
                                    targets=trg,
                                    pad_value=self.p.PAD) / self.p.grad_accum_steps

            # store accumulation until it is time to step
            accum_loss += loss
            accum_acc += acc
            self.grad_accum_step += 1

            if self.grad_accum_step % self.p.grad_accum_steps == 0:
                # perform step
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                            max_norm=self.p.max_abs_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.grad_accum_step = 0

                # update loss and acc sum and counter
                loss_counter += 1
                loss_sum += accum_loss
                acc_sum += accum_acc

                # reset accum values
                accum_loss = 0
                accum_acc = 0

            else:
                # only grad accum step, skip global_step increment at the end
                continue'''
            total_loss += loss.data[0]
            if (i + 1) % 100 == 0:
                loss_avg = total_loss / 100
                print('------------------------------------------')
                print('AVG LOSS UP TO NOW: ', loss_avg)
                total_loss = 0
            

    '''
    target_pad = 0
    optim.zero_grad()
    #here we compute the loss
    loss = F.cross_entropy(preds.view(-1, preds.size(-1)),
                results, ignore_index=target_pad)
    loss.backward()
    optim.step()

    #print things
    total_loss += loss.data[0]
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print("time = %dm, epoch %d, iter = %d, loss = %.3f,
                %ds per %d iters" % ((time.time() - start) // 60,
                epoch + 1, i + 1, loss_avg, time.time() - temp,
                print_every))
                total_loss = 0
                temp = time.time()

    #out = model(x, trg[:, :-1])
    #print(out.shape)'''

print('You are in the main file')