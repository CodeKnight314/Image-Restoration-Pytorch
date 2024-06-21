import torch
import torch.nn as nn
from base_model import BaseModelIR
from typing import Tuple, float

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
        self.fc1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=hidden_dim)
        self.fc2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)

        x = self.dw_conv(x)
        x = self.act(x)

        x = self.fc2(x)
        x = self.dropout(x)
        
        return x
    
class LeWinTransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, window_size, hidden_dim):
        super(LeWinTransformerBlock, self).__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.w_msa = W_MSA(channels, num_heads, window_size)
        self.leff = LeFF(channels, hidden_dim)
        self.skip_conv = nn.Conv2d(channels, hidden_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.w_msa(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.leff(x)
        x = x + self.skip_conv(shortcut)

        return x
    
class Downsample(nn.Module): 
    def __init__(self, channels): 
        super().__init__() 

        self.conv = nn.Conv2d(channels, channels * 2, kernel_size=4, stride=2, padding=1)

    def forward(self, x): 
        output = self.conv(x)
        return output

class Upsample(nn.Module): 
    def __init__(self, channels): 
        super().__init__() 

        self.conv = nn.ConvTranspose2d(channels, channels // 2, kernel_size=4, stride=2, padding=1)

    def forward(self, x): 
        output = self.conv(x)
        return output
    
class UFormer(BaseModelIR):
    def __init__(self, input_channels, output_channels, hidden_channels, depths: Tuple[int], heads: Tuple[int]):
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i in range(4):
            self.encoders.append(nn.Sequential(*[LeWinTransformerBlock(hidden_channels * 2 ** i, heads[i], 8, hidden_channels * 2 ** i)]))
            self.decoders.insert(0, nn.Sequential(*[LeWinTransformerBlock(hidden_channels * 2 ** (i+1), heads[i], 8, hidden_channels * 2 ** i)]))
            self.downsamples.append(Downsample(hidden_channels * 2 ** i))
            self.upsamples.insert(0, Upsample(hidden_channels * 2 ** (i+1)))

        self.bottleneck = nn.Sequential(*[LeWinTransformerBlock(hidden_channels * 2 ** 4, heads[-1], 8, hidden_channels * 2 ** 4) for _ in range(depths[-1])])

        self.output_conv = nn.Conv2d(hidden_channels, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x

        skips = []

        x = self.input_conv(x)

        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            skips.insert(0, x)
            x = self.downsamples[i](x)

        x = self.bottleneck(x)

        for i, decoder in enumerate(self.decoders):
            x = self.upsamples[i](x)
            x = torch.cat((x, skips[i]), dim=1)
            x = decoder(x)

        x = self.output_conv(x)
        output = x + residual

        return output
