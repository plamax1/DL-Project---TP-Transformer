import argparse
import importlib
import sys
import torch

import torch.nn as nn

###File just for TESTING
from tp-transformer import build_transformer
print("Building model ...")
##define build_transformer in model, and fix the file import
model = build_transformer("parameters to pass")
#and here we have the model

############ Define parameters
learning_rate = 0.01
beta1=1
beta2=2 #to see from the github repo


# Optimzier (Adam is the only option for now)
optimizer = torch.optim.Adam(params=model.parameters(),
                               lr=learning_rate,
                               betas=(beta1, beta2))
criterion = nn.CrossEntropyLoss(ignore_index=p.PAD)

# create a trainer object that will deal with all the training??

############ Dataset
  train_iterator = train_module.iterator
  eval_iterator = eval_module.iterator

########## Training

model.train() #put the model in training state
optimizer.zero_grad()
#batch traing
for idx, batch in enumerate(train_iterator):
      src = batch.src  # [batch_size, src_seq_len]
      trg = batch.trg  # [batch_size, trg_seq_len]
    #send data to device
      src = src.to(self.p.device)
      trg = trg.to(self.p.device)

      output = self.model(src, trg[:, :-1])
      # [batch_size, trg_seq_len-1, output_dim]

      flat_logits = logits.contiguous().view(-1, logits.shape[-1])
      # [batch_size * (trg_seq_len-1), output_dim]

      # ignore SOS symbol (skip first)
      flat_trg = trg[:, 1:].contiguous().view(-1)
      # [batch_size * (trg_seq_len-1)]

      # compute loss
      loss = self.criterion(flat_logits, flat_trg) / self.p.grad_accum_steps
      loss.backward()

      # compute acc
      acc = compute_accuracy(logits=logits,
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
        continue

      if self.global_step % self.p.log_every == 0 and idx != 0 \
              and self.global_step != 0:
        # compute loggin metrics
        elapsed = time.time() - start_time
        steps_per_sec = loss_counter / elapsed
        start_time = time.time()
        avg_loss = loss_sum / loss_counter
        avg_acc = acc_sum / loss_counter
        loss_sum, acc_sum, loss_counter = 0, 0, 0

       

      self.global_step += 1
print("doing initial evaluate()...")
