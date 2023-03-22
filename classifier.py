print('Classifier')

import torch
import torch.nn as nn
import pytorch_lightning as pl


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
        x = self.add_padding(train_batch[0], 200)
        y= self.add_padding(train_batch[1],35)
        logits = self.forward(x)
        loss = self.nllloss(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def test_step(self, test_batch,):
        x = self.add_padding(test_batch[0], 200)
        y= self.add_padding(test_batch[1],35)
        logits = self.forward(x)
        loss = self.nllloss(logits, y)
        prediction = torch.argmax(logits, dim=1)
        accuracy = torch.sum(y == prediction).item() / (len(y) * 1.0)
        output = dict({
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy),
        })
        return output
        

