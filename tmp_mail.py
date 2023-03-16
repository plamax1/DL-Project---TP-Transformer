from tp_transformer import *
from test_dataset_loading import *
import torch
import time
#import test_dataset_loading

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
    train_iter = get_demo_trainer(3)
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

    for epoch in range(epochs):
        for i, batch in enumerate(train_iter): #l'enumerate finisce non va avanti all'infinito
            src = batch[0]
            trg = batch[1]
            #print(batch[0].shape)
            #print(batch[1].shape)
            print(i)
            

    #in model si chama la forward di transformer
            preds = model(src,trg)
            
            print('PREDICTED: ', preds)
            print('PREDICTED SHAPE: ', preds.shape)
            print('TARGET SHAPE: ', trg.shape)
            print('Test.................')
            a=torch.flatten(preds)
            print('Flatten pred len: ', len(a))
            print('Sum of all the value in predit: ', torch.sum(a))


            #prediction shape: [batch_size, seq_lenght, trg_vocab_size]
            break

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