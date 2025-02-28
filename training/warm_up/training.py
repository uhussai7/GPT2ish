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
# num_epochs = 100
# model.train()

# for epoch in range(num_epochs):
#     total_loss = 0
#     for x, y, attention_mask in dataloader:
#         x, y, attention_mask = x.cuda(), y.cuda(), attention_mask.cuda()
#         optimizer.zero_grad()
#         outputs = model(x)#, mask=attention_mask)
#         loss = loss_fn(outputs.view(-1, outputs.size(-1)), y.reshape(-1))
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")


# model_save_path = "gpt2_tinystories.pth"
# torch.save(model.state_dict(), model_save_path)
# print(f"Model saved to {model_save_path}")

# Load the model
model_save_path = "gpt2_tinystories.pth"
loaded_model = GPT2(dm=dm, dk=dk, N=N, h=h, vocab_size=vocab_size, n_layers=n_layers)
loaded_model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
loaded_model = loaded_model.cuda() if torch.cuda.is_available() else loaded_model
loaded_model.eval()  # Important: Set the model to evaluation mode
print(f"Model loaded from {model_save_path}")


# Text generation
def generate_text(model, tokenizer, prompt, max_length=120):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    generated = input_ids
    for _ in range(max_length):
        with torch.no_grad():
            logits = model(generated)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        generated = torch.cat((generated, next_token), dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated[0], skip_special_tokens=True)

prompt = "There was this man you see"
print("Generated story:", generate_text(loaded_model, tokenizer, prompt,100))


# import torch
# from torch import nn, optim
# from modules.transformers import GPT2
# from transformers import GPT2Tokenizer, GPT2Model
# from datasets import load_dataset
# from torch.utils.data import Dataset, DataLoader


# dataset = load_dataset("roneneldan/TinyStories", split="train")  # Using 'train' split

# # # Step 2: Load GPT-2 tokenizer and model
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have an official pad token

# # Tokenize function
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# # Apply tokenization
# tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# # Convert to PyTorch tensors
# tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])


# class TinyStoriesDataset(Dataset):
#     def __init__(self, tokenized_dataset):
#         self.data = tokenized_dataset

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         input_ids = self.data[idx]["input_ids"]
#         attention_mask = self.data[idx]["attention_mask"]
#         return input_ids, attention_mask

# # Create DataLoader
# train_dataset = TinyStoriesDataset(tokenized_dataset)
# dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)











# # === Simple Dataset ===
# class SimpleLanguageDataset(Dataset):
#     def __init__(self, text, seq_length, vocab):
#         self.vocab = vocab
#         self.seq_length = seq_length
#         self.data = [self.vocab[char] for char in text]

#     def __len__(self):
#         return len(self.data) - self.seq_length

#     def __getitem__(self, idx):
#         x = torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long)
#         y = torch.tensor(self.data[idx + 1:idx + 1 + self.seq_length], dtype=torch.long)
#         return x, y

# # === Toy Data ===
# text = "hello hello hello hello"
# vocab = {'h':0, 'e':1, 'l':2, 'o':3, ' ':4}
# seq_length = 5
# dataset = SimpleLanguageDataset(text, seq_length, vocab)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# # === Model Setup ===
# dm, dk, N, h = 2, 2, seq_length*10, 2
# model = GPT2(dm, dk, N, h, vocab_size=len(vocab), n_layers=3)
# model = model if torch.cuda.is_available() else model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # === Optimizer & Loss ===
# optimizer = optim.Adam(model.parameters(), lr=3e-4)
# criterion = nn.CrossEntropyLoss()

# # === Training Loop ===
# epochs =500
# for epoch in range(epochs):
#     total_loss = 0
#     for x, y in dataloader:
#         x, y = x.to(device), y.to(device)
#         logits = model(x)  # [B, N, vocab_size]`~`

#         # Reshape for loss: (B*N, vocab_size) and (B*N)
#         loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

# print("✅ Training complete!")


# def generate_text(model, start_text, vocab, max_length=3):
#     idx_to_char = {idx: char for char, idx in vocab.items()}
#     input_ids = torch.tensor([vocab[c] for c in start_text], dtype=torch.long).unsqueeze(0)
#     input_ids = input_ids.to(next(model.parameters()).device)

#     model.eval()
#     with torch.no_grad():
#         for _ in range(max_length):
#             logits = model(input_ids)  # [B, N, vocab_size]
#             next_token_logits = logits[:, -1, :]  # Get the last token's logits
#             probs = torch.softmax(next_token_logits, dim=-1)
#             next_token = torch.multinomial(probs, num_samples=1)  # Sample from distribution
#             input_ids = torch.cat([input_ids, next_token], dim=1)

#     output_text = ''.join([idx_to_char[idx.item()] for idx in input_ids[0]])
#     return output_text

# # === Generate after training ===
# generated = generate_text(model, "hel", vocab,30)
# print(f"Generated text: {generated}")












# import torch
# from datasets import load_dataset
# from transformers import GPT2Tokenizer, GPT2Model
# from torch.utils.data import Dataset, DataLoader

# # Step 1: Load Tiny Stories dataset
# dataset = load_dataset("roneneldan/TinyStories", split="train")  # Using 'train' split

# # Step 2: Load GPT-2 tokenizer and model
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have an official pad token

# model = GPT2Model.from_pretrained("gpt2")
# model.eval()  # Evaluation mode

# # Step 3: Custom PyTorch Dataset
# class TinyStoriesDataset(Dataset):
#     def __init__(self, data, tokenizer, max_length=128):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         text = self.data[idx]['text']
#         tokenized = self.tokenizer(
#             text,
#             truncation=True,
#             max_length=self.max_length,
#             padding='max_length',
#             return_tensors='pt'
#         )
#         return tokenized['input_ids'].squeeze(0)  # (seq_len,)

# # Step 4: Create Dataset and DataLoader
# tiny_dataset = TinyStoriesDataset(dataset, tokenizer)
# data_loader = DataLoader(tiny_dataset, batch_size=8, shuffle=True)

# #get the embedding layer
# embedding_layer = model.get_input_embeddings()


# # Step 5: Extract first-layer embeddings (wte)
# for batch in data_loader:
#     print(batch.shape)

#     embeddings = embedding_layer(batch)
#     print(embeddings.shape)
#     print(embedding_layer.weight.shape)
    

#     cos_sim = torch.nn.functional.cosine_similarity(embeddings[:, :, None, :], embedding_layer.weight.shape[None, None, :, :], dim=-1)  # (1, seq_len, vocab_size)

#     recovered_ids = cos_sim.argmax(dim=-1)  # (1, seq_len)

#     # Decode back to text
#     recovered_text = tokenizer.batch_decode(recovered_ids, skip_special_tokens=True)
#     print("Recovered text:", recovered_text[0]) 


#     # with torch.no_grad():
#     #     first_layer_embeddings = model.transformer.wte(batch)  # (batch_size, seq_len, hidden_size)
#     # print(f"First layer embeddings shape: {first_layer_embeddings.shape}")
#     break  # Remove 'break' if processing all batches



