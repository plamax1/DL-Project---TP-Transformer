import argparse
import importlib
import sys
import torch

import torch.nn as nn

###File just for TESTING
from tp-transformer import build_transformer
print("Building model ...")
##define build_transformer in model, and fix the file import
build_transformer("parameters to pass")
#Define parameters
learning_rate = 0.01
beta1=1
beta2=2 #to see from the github repo


# Optimzier (Adam is the only option for now)
optimizer = torch.optim.Adam(params=model.parameters(),
                               lr=learning_rate,
                               betas=(beta1, beta2))
criterion = nn.CrossEntropyLoss(ignore_index=p.PAD)

# create a trainer object that will deal with all the training??
trainer = BasicSeq2SeqTrainer(model=model,
                              params=p,
                              train_iterator=train_iterator,
                              eval_iterator=eval_iterator,
                              optimizer=optimizer,
                              criterion=criterion,
                              log=log)


print("doing initial evaluate()...")
trainer.evaluate()