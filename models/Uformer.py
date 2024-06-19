import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class W_MSA(nn.Module):
    def __init__(self, channels, num_heads, window_size):
        super(W_MSA, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.d_k = channels // num_heads

        self.Q = nn.Conv2d(channels, self.d_k * self.num_heads, kernel_size=1, stride=1, padding=0)
        self.K = nn.Conv2d(channels, self.d_k * self.num_heads, kernel_size=1, stride=1, padding=0)
        self.V = nn.Conv2d(channels, self.d_k * self.num_heads, kernel_size=1, stride=1, padding=0)
        self.proj = nn.Conv2d(self.d_k * self.num_heads, channels, kernel_size=1, stride=1, padding=0)
        self.attn_drop = nn.Dropout(0.1)
        self.proj_drop = nn.Dropout(0.1)

    def window_partition(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, self.window_size ** 2, C)
        return x

    def window_reverse(self, x, H, W):
        B = int(x.shape[0] / (H * W / self.window_size ** 2))
        x = x.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
        return x

    def forward(self, x):
        B, C, H, W = x.shape 
        x = self.window_partition(x)

        Q = self.Q(x).reshape(-1, self.num_heads, self.window_size ** 2, C // self.num_heads)
        K = self.K(x).reshape(-1, self.num_heads, self.window_size ** 2, C // self.num_heads)
        V = self.V(x).reshape(-1, self.num_heads, self.window_size ** 2, C // self.num_heads)

        attn_output = self.attn_map(Q, K, V)

        attn_output = attn_output.reshape(-1, self.window_size ** 2, C)
        x = self.window_reverse(attn_output, H, W)

        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

    def attn_map(self, Q, K, V):
        attn_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        QK_prob = torch.softmax(attn_score, dim=-1)
        QK_prob = self.attn_drop(QK_prob)
        output = torch.matmul(QK_prob, V)
        return output
    
class LeFF(nn.Module):
    def __init__(self, in_channels, hidden_dim, kernel_size=3):
        super(LeFF, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, H, W, C = x.shape 

        x = x.view(B * H * W, C)
        x = self.fc1(x)
        x = self.act(x)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)  

        x = self.dw_conv(x)
        x = self.act(x)
        x = x.permute(0, 2, 3, 1).view(B * H * W, -1)  

        x = self.fc2(x)
        x = self.dropout(x)
        x = x.view(B, H, W, C)
        
        return x