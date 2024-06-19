import torch 
import torch.nn as nn
import math

class W_MSA(nn.Module): 
    def __init__(self, channels, num_heads, window_size, d_model):
        super().__init__() 

        self.num_heads = num_heads

        self.w_size = window_size

        self.d_k = channels // num_heads

        self.Q = nn.Conv2d(channels, self.d_k * self.num_heads, kernel_size=1, stride=1, padding=0)
        self.K = nn.Conv2d(channels, self.d_k * self.num_heads, kernel_size=1, stride=1, padding=0)
        self.V = nn.Conv2d(channels, self.d_k * self.num_heads, kernel_size=1, stride=1, padding=0)

    def attn_map(self, Q, K, V): 
        attn_score = torch.matmul(Q, torch.transpose(K, -2, -1)) / math.sqrt(self.d_k)
        QK_prob = torch.softmax(attn_score)
        output = torch.matmul(QK_prob, V)
        return output
    
    