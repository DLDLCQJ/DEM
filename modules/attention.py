import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.cbp_linear import CBPLinear

class CrossAttention(nn.Module): 
    def __init__(self, embed_dim, embed_dim_,num_heads=8, dropout=0.5):
        super(CrossAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim_, embed_dim_)
        self.v_linear = nn.Linear(embed_dim_, embed_dim_)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
         
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5 
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
  
    def forward(self, x1, x2):
        bs1 = x1.size(0)
        bs2 = x2.size(0)
        # Perform linear operation and split into heads
        Q = self.q_linear(x1).view(-1, bs1, self.num_heads, self.head_dim)
        K = self.k_linear(x2).view(-1, bs2, self.num_heads, self.head_dim)
        V = self.v_linear(x2).view(-1, bs2, self.num_heads, self.head_dim)

        # Scaled dot-product attention
        if Q.size(0) == K.size(0):
            attn_weights = torch.einsum('bnhd,bmhd->bnmh', Q, K) * self.scale
        else:
            print('mismatched source-target batchsize')
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.einsum('bnmh,bmhd->bnhd', attn_probs, V)
        attn_output = attn_output.reshape(bs1, self.embed_dim)
        
        attn_output = self.out_linear(attn_output)

        # Residual connection and layer normalization
        attn_output = self.norm1(x1 + attn_output)
        # Feed-forward network with residual connection and layer normalization
        ffn_output = self.ffn(attn_output)
        output = self.norm2(attn_output + ffn_output)

        return output

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.5):
        super(SelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
 
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
  
    def forward(self, x):
        bs = x.size(0)
   
        # Perform linear operation and split into heads
        Q = self.q_linear(x).view(-1, bs, self.num_heads, self.head_dim)
        K = self.k_linear(x).view(-1, bs, self.num_heads, self.head_dim)
        V = self.v_linear(x).view(-1, bs, self.num_heads, self.head_dim)

        # Scaled dot-product attention
        if Q.size(0) == K.size(0):
            attn_weights = torch.einsum('bnhd,bmhd->bnmh', Q, K) * self.scale
        else:
            print('mismatched source-target batchsize')
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.einsum('bnmh,bmhd->bnhd', attn_probs, V)
        attn_output = attn_output.reshape(bs, self.embed_dim)
        
        attn_output = self.out_linear(attn_output)

        # Residual connection and layer normalization
        attn_output = self.norm1(x + attn_output)
        # Feed-forward network with residual connection and layer normalization
        ffn_output = self.ffn(attn_output)
        output = self.norm2(attn_output + ffn_output)

        return output
    
class SelfAttention_cbp(nn.Module):
    def __init__(self, embed_dim, replacement_rate=1e-3,init='kaiming',maturity_threshold=10,num_heads=8,dropout=0.5):
        super(SelfAttention_cbp, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads 
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        # Feed-forward network 
        self.act=nn.ReLU()
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn1 = nn.Linear(embed_dim, embed_dim * 4)
        self.ffn2 = nn.Linear(embed_dim * 4, embed_dim)
        #self.cbp_ffn = CBPLinear(in_layer=self.ffn1, out_layer=self.ffn2, replacement_rate=replacement_rate, maturity_threshold=maturity_threshold, init=init)
        self.layers = nn.ModuleList([
            self.ffn1,self.act,self.dropout,self.ffn2
            ])
    def forward(self, x):
        bs = x.size(0)
   
        # Perform linear operation and split into heads
        Q = self.q_linear(x).view(-1, bs, self.num_heads, self.head_dim)
        K = self.k_linear(x).view(-1, bs, self.num_heads, self.head_dim)
        V = self.v_linear(x).view(-1, bs, self.num_heads, self.head_dim)

        # Scaled dot-product attention
        if Q.size(0) == K.size(0):
            attn_weights = torch.einsum('bnhd,bmhd->bnmh', Q, K) * self.scale
        else:
            print('mismatched source-target batchsize')

        attn_probs = self.softmax(attn_weights)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.einsum('bnmh,bmhd->bnhd', attn_probs, V)
        attn_output = attn_output.reshape(bs, self.embed_dim)

        attn_output = self.out_linear(attn_output)
        attn_output = self.norm1(x + attn_output)
        #
        ffn_output1 = self.layers[1](self.layers[0](attn_output))
        ffn_output2 = self.layers[3](self.layers[2](ffn_output1))
        output = self.norm2(attn_output + ffn_output2)

        return output,[ffn_output1]

class CrossAttention_cbp(nn.Module): 
    def __init__(self, embed_dim, embed_dim_,replacement_rate=1e-3,init='kaiming',maturity_threshold=10,num_heads=8,dropout=0.5):
        super(CrossAttention_cbp, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim_, embed_dim_)
        self.v_linear = nn.Linear(embed_dim_, embed_dim_)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
          
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5 
        # Feed-forward network
        self.act=nn.ReLU()
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn1 = nn.Linear(embed_dim, embed_dim * 4)
        self.ffn2 = nn.Linear(embed_dim * 4, embed_dim)

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([
            self.ffn1,self.act,self.dropout,self.ffn2
            ])
    def forward(self, x1, x2):
        bs1 = x1.size(0)
        bs2 = x2.size(0)
        # Perform linear operation and split into heads
        Q = self.q_linear(x1).view(-1, bs1, self.num_heads, self.head_dim)
        K = self.k_linear(x2).view(-1, bs2, self.num_heads, self.head_dim)
        V = self.v_linear(x2).view(-1, bs2, self.num_heads, self.head_dim)

        # Scaled dot-product attention
        if Q.size(0) == K.size(0):
            attn_weights = torch.einsum('bnhd,bmhd->bnmh', Q, K) * self.scale
        else:
            print('mismatched source-target batchsize')
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.einsum('bnmh,bmhd->bnhd', attn_probs, V)
        attn_output = attn_output.reshape(bs1, self.embed_dim)
        
        attn_output = self.out_linear(attn_output)

        # Residual connection and layer normalization
        attn_output = self.norm1(x1 + attn_output)
        # Feed-forward network with residual connection and layer normalization
        ffn_output1 = self.layers[1](self.layers[0](attn_output))
        ffn_output2 = self.layers[3](self.layers[2](ffn_output1))
        output = self.norm2(attn_output + ffn_output2)

        return output,[ffn_output1]

# class SelfAttention_cbp(nn.Module):
#     def __init__(self, embed_dim, replacement_rate=1e-3,init='kaiming',maturity_threshold=10,num_heads=8,dropout=0.5):
#         super(SelfAttention_cbp, self).__init__()
#         assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = self.embed_dim // self.num_heads
        
#         self.q_linear = nn.Linear(embed_dim, embed_dim)
#         self.k_linear = nn.Linear(embed_dim, embed_dim)
#         self.v_linear = nn.Linear(embed_dim, embed_dim)
#         self.out_linear = nn.Linear(embed_dim, embed_dim)
#         self.norm1 = nn.LayerNorm(embed_dim)
        
#         self.softmax = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(dropout)
#         self.scale = self.head_dim ** -0.5
#         # Feed-forward network
#         self.act=nn.ReLU()
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.ffn1 = nn.Linear(embed_dim, embed_dim * 4)
#         self.ffn2 = nn.Linear(embed_dim * 4, embed_dim)
#         self.cbp_ffn = CBPLinear(in_layer=self.ffn1, out_layer=self.ffn2, replacement_rate=replacement_rate, maturity_threshold=maturity_threshold, init=init)
#         self.layers = nn.ModuleList([
#             self.ffn1,self.act,self.dropout,self.ffn2
#             ])
        
  
#     def forward(self, x):
#         bs = x.size(0)
   
#         # Perform linear operation and split into heads
#         Q = self.q_linear(x).view(-1, bs, self.num_heads, self.head_dim)
#         K = self.k_linear(x).view(-1, bs, self.num_heads, self.head_dim)
#         V = self.v_linear(x).view(-1, bs, self.num_heads, self.head_dim)

#         # Scaled dot-product attention
#         if Q.size(0) == K.size(0):
#             attn_weights = torch.einsum('bnhd,bmhd->bnmh', Q, K) * self.scale
#         else:
#             print('mismatched source-target batchsize')

#         attn_probs = self.softmax(attn_weights)
#         attn_probs = self.dropout(attn_probs)
#         attn_output = torch.einsum('bnmh,bmhd->bnhd', attn_probs, V)
#         attn_output = attn_output.reshape(bs, self.embed_dim)

#         attn_output = self.out_linear(attn_output)
#         attn_output = self.norm1(x + attn_output)
#         #
#         ffn_output1 = self.cbp_ffn(self.act(self.ffn1(attn_output)))
#         ffn_output2 = self.ffn2(self.dropout(ffn_output1))
#         output = self.norm2(attn_output + ffn_output2)

#         return output,[ffn_output1]

