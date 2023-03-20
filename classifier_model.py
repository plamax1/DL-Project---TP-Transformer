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
        
    def forward(self, x):
        h1 = self.act1(self.input(x))
        h2 = self.act2(self.input(h1))
        out= h2.split(self.vocab_size)
        result = []
        for i in out:
            result.append(torch.argmax(self.sotfmax(i)))
            
        return h2
    

