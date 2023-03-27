# Deep Learning Project 2022-23
## Question Answering on Mathematics Dataset. 
- Sota implementation : Tp- Transformer
- Other baseline implementation: Transformer
- Trivial implementation: Simple NN

## Requirements
```bash
pip install pandaraller pytorch pytorch-lightning
```
## Usage:
```bash
python main.py --mode <mode> --model <model> --batch_size <bts> --epochs <n> --train_pct <pct> --test_pct <pct> --model_name <model_name>
```

## Colab link
```bash
https://colab.research.google.com/drive/1KKbsHyVp7yfKSh9meXU3SZlPH_9BkGz5#scrollTo=eEG-73l8abXg
```

## Pretrained models
### Tp-transformer
```bash
https://drive.google.com/file/d/17nswP_HX9nZtg7ilQWQzTD1NVPH0P07v/view?usp=sharing
```
### Transformer
```bash
https://drive.google.com/file/d/1RaQi5wSuOXp0Bh9aFa0XDfkhwZ3n2Ta9/view?usp=sharing
```
### Trivial Baseline
```bash
https://drive.google.com/file/d/1kYYXZkjCPRAb6-t1jSSbmAJzyyYyEkt8/view?usp=sharing
```

### Test mode
```bash
!python main.py --mode train --model Classifier --batch_size 256 --epochs 2 --train_pct 0.005 --test_pct 0.1
```

### Eval mode
```bash
!python main.py --mode load_eval --model <model> --test_pct 0.1 
```


### Accuracy Results
| Attempt | Tp-transformer    | Transformer    | Trivial baseline
| :---:   | :---: | :---: | :---: |
| Interpolate | 8.37   | 8.36   | 2.26
| Extrapolate | 10.58   | 10.58   | 2.36

### Loss Results
| Attempt | Tp-transformer    | Transformer    | Trivial baseline
| :---:   | :---: | :---: | :---: |
| Interpolate | 3.31   | 3.30   |-0.27
| Extrapolate | 3.31  | 3.30   |-0.22
