import sys
import os 

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from modules.transformers import SelfAttention
import torch

attn=SelfAttention(4,4,256)
X_in=torch.rand([10,256,4])
print(attn(X_in)[0].shape)

from modules.transformers import GPT2

gpt2=GPT2(4,4,100,1,50,2)

