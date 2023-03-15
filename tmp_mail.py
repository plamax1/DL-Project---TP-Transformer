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
    src_vocab_size = 200
    trg_vocab_size = 10
    
    #Create model
    print('Creating model...')
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
        device
    )

    ### Dataset loading
    train_iter = get_demo_trainer(32)
    print('Model created')
    #Now how to train the model?
    print('Starting model training')
    #Defining optimizer
    optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    model.train()
    start = time.time()
    temp = start
    total_loss = 0
    epochs = 10

    for i in range(epochs):
        for i, batch in enumerate(train_iter):
            src = batch[0]
            trg = batch[1]
            #print(batch[0].shape)
            #print(batch[1].shape)



    #out = model(x, trg[:, :-1])
    #print(out.shape)'''

    print('You are in the main file')