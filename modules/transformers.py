import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from torch import nn, optim


class SelfAttention(nn.Module):
    def __init__(self, dm, dk, N_max, h=1):
        super(SelfAttention, self).__init__()
        self.dm = dm    # Model dimension
        self.dk = dk    # Key/Query dimension per head
        self.h = h      # Number of heads
        self.N_max = N_max      # Max Sequence length

        # Linear layer to project input into Q, K, V
        self.Wqkv = nn.Linear(dm, 3 * dk * h)
        
        # Final linear layer to project concatenated heads back to dm
        self.final = nn.Linear(h * dk, dm)

        # Causal mask to prevent attention to future tokens
        self.register_buffer("causal_mask", self.create_causal_mask(N_max))

        # Weight initialization (similar to GPT-2)
        nn.init.normal_(self.Wqkv.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.final.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None):
        B, N, _ = x.shape  # Batch size, Sequence length, Model dimension

        # Linear projection to obtain Q, K, V
        QKV = self.Wqkv(x)
        Q, K, V = torch.split(QKV, [self.dk * self.h] * 3, dim=-1)

        # Reshape Q, K, V to [B, h, N, dk]
        Q, K, V = map(self.reshape, (Q, K, V))

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dk ** 0.5)

        # Apply causal mask
        #causal_mask = torch.triu(torch.full((N, N), float('-inf')), diagonal=1).to(attn_scores.device)
        attn_scores = attn_scores + self.causal_mask[None, None, :N, :N]

        # Apply external mask if provided (optional)
        if mask is not None:
            attn_scores += mask[:, None, :, :].masked_fill(mask == 0, float('-inf'))

        # Softmax to obtain attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Attention output
        attn_output = torch.matmul(attn_weights, V)  # [B, h, N, dk]

        # Concatenate heads and project back to dm
        attn_output = attn_output.transpose(1, 2).reshape(B, N, self.h * self.dk)
        output = self.final(attn_output)

        return output

    def reshape(self, A):
        """
        Reshape input from [B, N, h * dk] to [B, h, N, dk].
        """
        B, N, _ = A.shape
        return A.view(B, N, self.h, self.dk).permute(0, 2, 1, 3)

    def create_causal_mask(self, N):
        """
        Creates a causal mask to prevent attention to future tokens.
        Shape: [N, N] with -inf in the upper triangle (excluding diagonal).
        """
        mask = torch.triu(torch.full((N, N), float('-inf')), diagonal=1)
        return mask
    
class GPT2(nn.Module):
    def __init__(self, dm, dk, N, h=1, vocab_size=50000, n_layers=2):
        super(GPT2,self).__init__()

        self.dm = dm
        self.dk = dk
        self.N = N
        self.h = h
        self.vocab_size=vocab_size
        self.n_layers=n_layers

        self.embeddings = nn.Embedding(self.vocab_size,self.dm)

        self.position_embeddings= nn.Embedding(self.N,self.dm)

        self.blocks = nn.ModuleList([Block(dm,dk,N,h) for _ in range(0,self.n_layers)])
        
        self.lm_head = nn.Linear(self.dm,self.vocab_size,bias=False)


    def forward(self, input_ids, mask=None ):
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0)
        input_embeds = self.embeddings(input_ids) + self.position_embeddings(position_ids)

        hidden_states=input_embeds
        for block in self.blocks:
            hidden_states = block(hidden_states, mask)

        logits = self.lm_head(hidden_states)
        return logits


class Block(nn.Module):
    def __init__(self, dm, dk, N, h=1):
        super(Block,self).__init__()
        self.dm = dm
        self.dk = dk
        self.N = N
        self.h = h

        self.attn = SelfAttention(dm,dk,N,h)
        self.lffn = LastFeedForward(dm)

        self.ln1 = nn.LayerNorm(dm)
        self.ln2 = nn.LayerNorm(dm)

    def forward(self,x, mask=None):

        #Attentin block
        attn_x = self.attn(x, mask)
        x = self.ln1(x + attn_x)

        #feed forward block
        ffn_x = self.lffn(x)
        x = self.ln2(x + ffn_x)
        return x

class LastFeedForward(nn.Module):
    def __init__(self,dm,scale_up=4):
        super(LastFeedForward,self).__init__()

        self.scale_up = scale_up
        self.dm = dm
        self.fc1=nn.Linear(dm,dm*self.scale_up)
        self.fc2=nn.Linear(dm*self.scale_up, dm)

    def forward(self,x):
        return self.fc2(F.gelu(self.fc1(x)))
    

class GPT2Lightning(L.LightningModule):
    def __init__(self, dm, dk, N, h=1, vocab_size=50000, n_layers=2, learning_rate=5e-5):
        super(GPT2Lightning, self).__init__()
        self.save_hyperparameters()
        self.model = GPT2(dm, dk, N, h, vocab_size, n_layers)
        self.learning_rate = learning_rate

    def forward(self, input_ids, mask=None):
        return self.model(input_ids, mask)

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        logits = self(input_ids)
        loss = F.cross_entropy(logits.view(-1, self.hparams.vocab_size), labels.view(-1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        logits = self(input_ids)
        loss = F.cross_entropy(logits.view(-1, self.hparams.vocab_size), labels.view(-1))
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)