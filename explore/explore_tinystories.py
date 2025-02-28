import sys
import os 

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


import torch
from torch import nn, optim
from modules.transformers import GPT2
from transformers import GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# Model, optimizer, and loss
dm=128
dk=16
N=128
h=8
n_layers =4

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
vocab_size=tokenizer.vocab_size
model = GPT2(dm=dm, dk=dk, N=N, h=h, vocab_size=vocab_size, n_layers=n_layers)
model = model.cuda() if torch.cuda.is_available() else model

# Model, optimizer, and loss
dm=128
dk=16
N=128
h=8
n_layers =4
vocab_size=tokenizer.vocab_size
model = GPT2(dm=dm, dk=dk, N=N, h=h, vocab_size=vocab_size, n_layers=n_layers)
model = model.cuda() if torch.cuda.is_available() else model


