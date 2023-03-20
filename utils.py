import torch
def compute_accuracy(logits, targets, pad_value):
  """
  Compute full sequence accuracy of a batch.
  :param logits: the model logits (batch_size, seq_len, out_dim)
  :param targets: the true targets (batch_size, seq_len)
  :param pad_value: PAD value used to fill end of target seqs
  :return: continous accuracy between 0.0 and 1.0
  """
  trg_shifted = targets[:, 1:]              # drop the SOS from targets
  y_hat = torch.argmax(logits, dim=-1)      # get index predictions from logits

  # count matches in batch, masking out pad values in each target
  matches = (torch.eq(trg_shifted,y_hat) | (trg_shifted==pad_value)).all(1).sum().item()
  
  acc_percent = matches / len(logits)
  return acc_percent

def preds_to_string(preds, voc):
  result=''
  for idx in preds:
    result+=voc.itos[idx]
  return result