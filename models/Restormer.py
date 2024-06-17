import torch 
import torch.nn as nn 
from base_model import BaseModelIR

class MDTA(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.num_heads = heads
        self.layer_norm = nn.LayerNorm(channels)
        self.scale_parameter = nn.Parameter(torch.ones(1))

        self.Q = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.K = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.V = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.O = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def attn_calculation(self, Q, K, V):
        attn_score = torch.matmul(K, Q) / self.scale_parameter
        QK_probs = torch.softmax(attn_score, dim=-1)
        output = torch.matmul(V, QK_probs)
        return output

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.layer_norm(x)

        Q = self.Q(x).reshape(B, self.num_heads, H * W, C // self.num_heads)
        K = self.K(x).reshape(B, self.num_heads, H * W, C // self.num_heads).permute(0, 1, 3, 2)
        V = self.V(x).reshape(B, self.num_heads, H * W, C // self.num_heads)

        attn_score = self.attn_calculation(Q, K, V)
        attn_score = attn_score.permute(0, 1, 3, 2).reshape(B, C, H, W)
        output = self.O(attn_score)
        return output

class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super().__init__()
        self.layer_norm = nn.LayerNorm(channels)
        self.gating_conv = nn.Sequential(
            nn.Conv2d(channels, channels * expansion_factor, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(channels * expansion_factor, channels * expansion_factor, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU()
        )
        self.information_conv = nn.Sequential(
            nn.Conv2d(channels, channels * expansion_factor, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(channels * expansion_factor, channels * expansion_factor, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.reconstruction_conv = nn.Conv2d(channels * expansion_factor, channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.layer_norm(x)
        gated_conv_output = self.gating_conv(x)
        information_conv_output = self.information_conv(x)
        information_gated = torch.mul(gated_conv_output, information_conv_output)
        output = self.reconstruction_conv(information_gated)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, channels, expansion_factor, num_heads):
        super().__init__()
        self.MDTA = MDTA(channels=channels, heads=num_heads)
        self.GDFN = GDFN(channels=channels, expansion_factor=expansion_factor)

    def forward(self, x):
        x = self.MDTA(x) + x
        x = self.GDFN(x) + x
        return x

class Restormer(BaseModelIR):
    def __init__(self, input_channels, output_channels, channels, num_levels, num_transformers, num_heads, expansion_factor):
        super().__init__()
        self.feature_extraction_conv = nn.Conv2d(input_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.levels = nn.ModuleList()

        for i in range(num_levels):
            self.levels.append(nn.Sequential(
                *[TransformerBlock(channels=channels[i],
                                   expansion_factor=expansion_factor,
                                   num_heads=num_heads[i]) for _ in range(num_transformers[i])]
            ))

        for i in range(num_levels - 2, -1, -1):
            self.levels.append(nn.Sequential(
                nn.Conv2d(channels[i+1] * 2, channels[i], kernel_size=1, stride=1, padding=0, bias=False),
                *[TransformerBlock(channels=channels[i],
                                   expansion_factor=expansion_factor,
                                   num_heads=num_heads[i]) for _ in range(num_transformers[i])]
            ))

        self.refinement = nn.Sequential(
            *[TransformerBlock(channels=channels[0], expansion_factor=expansion_factor, num_heads=num_heads[0]) for _ in range(num_transformers[0])]
        )

        self.feature_reconstruction = nn.Conv2d(channels[0], output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.upsample = nn.PixelShuffle(2)
        self.downsample = nn.PixelUnshuffle(2)

    def forward(self, x):
        residual = x
        x = self.feature_extraction_conv(x)
        skip_connections = []
        for i in range(len(self.levels) // 2):
            x = self.levels[i](x)
            skip_connections.append(x)
            x = self.downsample(x)

        x = self.levels[len(self.levels) // 2](x)

        for i in range(len(self.levels) // 2 + 1, len(self.levels)):
            x = torch.cat([self.upsample(x), skip_connections.pop()], dim=1)
            x = self.levels[i](x)

        x = self.refinement(x)
        x = self.feature_reconstruction(x)
        output = x + residual
        return output