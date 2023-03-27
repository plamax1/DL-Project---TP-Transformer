# from test_dataset_loading import *
from dataset_loading import Vocabulary, get_train_iterator
import torch
import pytorch_lightning as pl
import torch
import argparse
from classifier import Multiclass
from tp_transformer import TpTransformer
from transformer import Transformer


import os
import pathlib
path = pathlib.Path().resolve()
train_path = os.path.join(path, 'Dataset/Train/**/*.txt')
test_path_int = os.path.join(path, 'Dataset/Test/interpolate**/*.txt')
test_path_ext = os.path.join(path, 'Dataset/Test/extrapolate**/*.txt')
#demo_path = os.path.join(path, 'Demo/**/*.txt')
algebra_path = os.path.join(path, 'Algebra/**/*.txt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Question answerin on mathematic dataset')
    parser.add_argument('--mode', type=str, default=None,
                        help='usage mode: select load_eval or train (default: "")')
    parser.add_argument('--model', type=str, default=None,
                        help='choose among: Transformer, Tp-transformer or Classifier (default: "")')
    parser.add_argument('--model_name', type=str, default='model.pt',
                        help='path of the model you want to load (default: "model.pt")')
    parser.add_argument('--train_pct', type=float, default=1,
                        help='Insert the percentage of the train dataset you wanto to load [0,1] (default: "1")')
    parser.add_argument('--test_pct', type=float, default=1,
                        help='Insert the percentage of the test dataset you wanto to load [0,1] (default: "1")')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='the batch size for each step (default: "1024")')
    parser.add_argument('--epochs', type=int, default=10,
                        help='the number of epochs you want the trainer to run for (default: "10")')
    
    arguments = parser.parse_args()
    batch_size = arguments.batch_size
    epochs = arguments.epochs
    mode = arguments.mode
    model_name = arguments.model_name
    train_pct = arguments.train_pct
    test_pct = arguments.test_pct
    model_type = arguments.model
    if (not (mode == 'train' or mode == 'load_eval')):
        print('Only 2 modes available: train and load_eval')
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Defining parameters
    vocab_size = 73
    # Creating vocab
    voc = Vocabulary(vocab_size)  # 73 is the vocabulary len used in the paper
    # build vocab
    voc.build_vocabulary()
    print('VOCABULARY CREATED')
    # Create model
    print('Creating model...')
    if arguments.model == 'Classifier':
        model = Multiclass(200, 35, vocab_size)
    elif arguments.model == 'Tp-transformer':
        model = TpTransformer(vocab_size)
    elif arguments.model == 'Transformer':
        model = Transformer(vocab_size)
    else:
        print('Only 3 models available: Transformer, Tp-transformer, Classifier \n Mind the initial uppercase')
        exit(1)

    if (mode == 'load_eval'):
        model = torch.load(model_name)
        print('Model loaded succesfully: ', model_name)
        # print(model)
        trainer = pl.Trainer()
        print('Getting test iterator...')
        test_iterator_int = get_train_iterator(
                test_path_int, batch_size, voc, test_pct)
        test_iterator_ext = get_train_iterator(
                test_path_ext, batch_size, voc, test_pct)
        print('Starting evaluation of the model...')
        # Perform evaluation
        print('Testing on interpolate test Dataset...')
        print(trainer.test(model, dataloaders=test_iterator_int))
        print('Testing on extrapolate test Dataset...')
        print(trainer.test(model, dataloaders=test_iterator_ext))
    elif (mode == 'train'):
        trainer = pl.Trainer(max_epochs=epochs)
        print('Getting train iterator...')
        train_iterator = get_train_iterator(
            train_path, batch_size, voc, train_pct)
        print('Getting test iterator...')
        test_iterator_int = get_train_iterator(
            test_path_int, batch_size, voc, test_pct)
        test_iterator_ext = get_train_iterator(
            test_path_ext, batch_size, voc, test_pct)
        trainer.fit(model, train_dataloaders=train_iterator)
        print('Saving model...')
        torch.save(model, 'saved_' + model_type+'.pt')
        print('Model saved...')
        print('Starting evaluation of the model...')
        # Perform evaluation
        print('Testing on interpolate test Dataset...')
        print(trainer.test(model, dataloaders=test_iterator_int))
        print('Testing on extrapolate test Dataset...')
        print(trainer.test(model, dataloaders=test_iterator_ext))
    else:
        print('Only 2 modes available: train and load eval')
        exit(1)
