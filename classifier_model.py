print('Classifier')

import torch
import torch.nn as nn

class Multiclass(nn.Module,  ):
    def __init__(self, max_input_len, max_output_len, vocab_size):
        super().__init__()
        self.input = nn.Linear(max_input_len, 500)
        self.act1 = nn.ReLU()
        self.act2= nn.ReLU()
        self.hidden = nn.Linear(500, 2000)
        self.output = nn.Linear(2000, vocab_size*max_output_len)
        self.sotfmax = nn.Softmax()
        self.vocab_size = vocab_size
        
    def forward(self, data):
        batch_size = data.shape[0]
        x= data.view(batch_size, -1)
        h1 = self.act1(self.input(x))
        h2 = self.act2(self.hidden(h1))
        out=self.output(h2)
        print(type(out))
        out= out.reshape(10, -1, self.vocab_size)
        out2= torch.softmax(out, dim=2)
        print(type(out))
        tmp= out2.reshape(-1, self.vocab_size)
        #size [batch_size*max_len, vocab_size]
        result = []
        for i in tmp:
            result.append(torch.argmax(i))
        result= torch.stack(result).reshape(batch_size, -1)
        #[batch_size, max_output_len]

        print('results shape: ', result.shape)
            
        return result
    

