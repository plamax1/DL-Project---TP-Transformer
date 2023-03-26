# DL-Project---TP-Transformer
DL Project - TP Transformer
#Deep Learning Project 2022-23
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
python main.py --mode <mode> --model <model> --batch_size <bts> --epochs <n> --train_pct <pct> --test_pct <pct>
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
```

### Results
| Attempt | Tp-transformer    | Transformer    | Trivial baseline
| :---:   | :---: | :---: | :---: |
| Interpolate | 301   | 283   |
| Extrapolate | 301   | 283   |

