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

# Load dataset
dataset = load_dataset("roneneldan/TinyStories", split="train")

# Select a subset of the dataset (e.g., first 10,000 samples)
subset_size = 10000
subset_dataset = dataset.select(range(subset_size))

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = subset_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

class TinyStoriesDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.data = tokenized_dataset
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        input_ids = self.data[idx]["input_ids"]
        attention_mask = self.data[idx]["attention_mask"]
        x = input_ids[:-1]
        y = input_ids[1:]
        return x, y, attention_mask[:-1]

train_dataset = TinyStoriesDataset(tokenized_dataset)
dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Model, optimizer, and loss
dm=128
dk=16
N=128
h=8
n_layers =4
vocab_size=tokenizer.vocab_size
model = GPT2(dm=dm, dk=dk, N=N, h=h, vocab_size=vocab_size, n_layers=n_layers)
model = model.cuda() if torch.cuda.is_available() else model

optimizer = optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

# Training loop
num_epochs = 100
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for x, y, attention_mask in dataloader:
        x, y, attention_mask = x.cuda(), y.cuda(), attention_mask.cuda()
        optimizer.zero_grad()
        outputs = model(x)#, mask=attention_mask)
        loss = loss_fn(outputs.view(-1, outputs.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

model_save_path = "gpt2_tinystories.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")


