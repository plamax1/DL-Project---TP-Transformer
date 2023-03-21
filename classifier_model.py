print('Classifier')

import torch
import torch.nn as nn
import pytorch_lightning as pl

from dataset_loading import Vocabulary, get_train_iterator, tensor_to_string, get_test_iterator
    # Creating vocab
voc = Vocabulary(73) #73 is the vocabulary len used in the paper
    #build vocab
voc.build_vocabulary()
print('VOCABULARY CREATED')
print('Vocabulary lenght: ', len(voc))

class Multiclass(pl.LightningModule):
    def __init__(self, max_input_len, max_output_len, vocab_size):
        super().__init__()
        self.input = nn.Linear(max_input_len, 500)
        self.act1 = nn.ReLU()
        self.act2= nn.ReLU()
        self.hidden = nn.Linear(500, 2000)
        self.output = nn.Linear(2000, vocab_size*max_output_len)
        self.sotfmax = nn.Softmax()
        self.vocab_size = vocab_size

    def add_padding(self, data, max_lenght):
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
            
    def forward(self, data):
        batch_size = data.shape[0]
        x= data.view(batch_size, -1)
        h1 = self.act1(self.input(x))
        h2 = self.act2(self.hidden(h1))
        out=self.output(h2)
        out= out.reshape(batch_size, -1, self.vocab_size)
        out2= torch.softmax(out, dim=2)
        return out2
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def nllloss(self, logits, labels):
        #print('LOSS: logits shape: ', logits.shape)
        #print('LOSS: target shape: ', labels.shape)
        return nn.NLLLoss()(logits.reshape(-1, 73), torch.flatten(labels).long())

    def training_step(self, train_batch):
        #filelist = ['test.txt']
        #for file in filelist:
         #   print('Loading file : ', file)
          #  train_batch=get_train_iterator(file, 10, voc)
            x = self.add_padding(train_batch[0], 200)
            y= self.add_padding(train_batch[1],35)
            logits = self.forward(x)
            loss = self.nllloss(logits, y)
            self.log('train_loss', loss)
            return loss
    
    def validation_step(self, val_batch):
      x = val_batch[0]
      y= val_batch[1]
      logits = self.forward(x)
      loss = self.cross_entropy_loss(logits, y)
      self.log('val_loss', loss)
    

