import torch 
import torch.nn as nn
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

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.window_size == 0 and W % self.window_size == 0, "Input dimensions must be divisible by window size."

        x = x.view(B, C, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, self.window_size, self.window_size)
        
        Q = self.Q(x).view(-1, self.num_heads, self.d_k, self.window_size * self.window_size).permute(0, 1, 3, 2)
        K = self.K(x).view(-1, self.num_heads, self.d_k, self.window_size * self.window_size).permute(0, 1, 3, 2)
        V = self.V(x).view(-1, self.num_heads, self.d_k, self.window_size * self.window_size).permute(0, 1, 3, 2)

        attn_output = self.attn_map(Q, K, V)

        attn_output = attn_output.permute(0, 1, 3, 2).contiguous().view(-1, C, self.window_size, self.window_size)
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)

        attn_output = attn_output.view(B, H // self.window_size, W // self.window_size, C, self.window_size, self.window_size)
        attn_output = attn_output.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)

        return attn_output

    def attn_map(self, Q, K, V):
        attn_score = torch.matmul(Q, torch.transpose(K, -2, -1)) / math.sqrt(self.d_k)
        QK_prob = torch.softmax(attn_score, dim=-1)
        QK_prob = self.attn_drop(QK_prob)
        output = torch.matmul(QK_prob, V)
        return output
    
