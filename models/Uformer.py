import torch
import torch.nn as nn

class W_MSA(nn.Module):
    def __init__(self, channels, num_heads, window_size):
        super(W_MSA, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.d_k = channels // num_heads

        self.scale = (self.d_k) ** -0.5

        self.Q = nn.Conv2d(channels, self.d_k * self.num_heads, kernel_size=1, stride=1, padding=0)
        self.K = nn.Conv2d(channels, self.d_k * self.num_heads, kernel_size=1, stride=1, padding=0)
        self.V = nn.Conv2d(channels, self.d_k * self.num_heads, kernel_size=1, stride=1, padding=0)
        self.proj = nn.Conv2d(self.d_k * self.num_heads, channels, kernel_size=1, stride=1, padding=0)

    def window_partition(self, x): 
        B, C, H, W = x.shape 
        x = x.reshape(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous() 
        x = x.view(B, C, -1, self.window_size * self.window_size)
        return x

    def window_unpartition(self, x, H, W): 
        B, _, _, C = x.shape
        x = x.view(B, -1, self.window_size, self.window_size, C)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.reshape(B, C, H, W)
        return x

    def attn_map(self, Q, K, V): 
        QK_probs = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_score = torch.softmax(QK_probs, dim=-1)
        scaled_values = torch.matmul(attn_score, V)
        return scaled_values
    
    def forward(self, x): 
        B, C, H, W = x.shape
        partitioned_inputs = self.window_partition(x)

        Q = self.Q(partitioned_inputs).permute(0, 2, 3, 1).contiguous().reshape(B, self.num_heads, -1, self.window_size * self.window_size, self.d_k)
        K = self.K(partitioned_inputs).permute(0, 2, 3, 1).contiguous().reshape(B, self.num_heads, -1, self.window_size * self.window_size, self.d_k)
        V = self.V(partitioned_inputs).permute(0, 2, 3, 1).contiguous().reshape(B, self.num_heads, -1, self.window_size * self.window_size, self.d_k)

        attn_map = self.attn_map(Q, K, V).reshape(B, -1, self.window_size * self.window_size, self.d_k * self.num_heads)

        unpartioned_output = self.window_unpartition(attn_map, H, W)

        projection_output = self.proj(unpartioned_output)

        return projection_output

    
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
    
class LeWinTransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, window_size, hidden_dim):
        super(LeWinTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.w_msa = W_MSA(channels, num_heads, window_size)
        self.leff = LeFF(channels, hidden_dim)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.w_msa(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.leff(x)
        x = x + shortcut

        return x