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




#tokenization
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Model, optimizer, and loss
dm=128
dk=16
N=128
h=8
n_layers =4
vocab_size=tokenizer.vocab_size

# Load the model
model_save_path = "gpt2_tinystories.pth"
loaded_model = GPT2(dm=dm, dk=dk, N=N, h=h, vocab_size=vocab_size, n_layers=n_layers)
loaded_model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
loaded_model = loaded_model.cuda() if torch.cuda.is_available() else loaded_model
loaded_model.eval()  # Important: Set the model to evaluation mode
print(f"Model loaded from {model_save_path}")

# Probabilistic text generation
def generate_text(model, tokenizer, prompt, max_length=120, temperature=1.0):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    generated = input_ids

    for _ in range(max_length):
        with torch.no_grad():
            logits = model(generated)[:, -1, :]

        # Apply temperature scaling
        logits = logits / temperature

        # Apply Top-k and Top-p sampling
        probabilities = torch.softmax(logits, dim=-1)

        # Sample next token (instead of argmax)
        next_token = torch.multinomial(probabilities, num_samples=1)

        generated = torch.cat((generated, next_token), dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)

# Example
prompt = "There was this man you see"
print("Generated story:", generate_text(loaded_model, tokenizer, prompt, max_length=100, temperature=0.1))
