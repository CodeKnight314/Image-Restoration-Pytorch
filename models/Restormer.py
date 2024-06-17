import torch 
import torch.nn as nn 

class MDTA(nn.Module): 
    def __init__(self, channels, heads): 
        super().__init__() 

        self.num_heads = heads

        self.scale_parameter = nn.Parameter(torch.ones(1))

        self.Q = nn.Sequential(*[nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
                               nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)])
        
        self.K = nn.Sequential(*[nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
                               nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)])
        
        self.V = nn.Sequential(*[nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
                               nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)])
        
        self.O = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def attn_calculation(self, Q, K, V):
        attn_score = torch.matmul(K, Q) / self.scale_parameter
        QK_probs = torch.softmax(attn_score, dim=-1)
        output = torch.matmul(V, QK_probs)
        return output

    def forward(self, x): 
        B, C, H, W = x.shape
        
        Q = self.Q(x).reshape(B, self.num_heads, H * W, C // self.num_heads)
        K = self.K(x).reshape(B, self.num_heads, H * W, C // self.num_heads).permute(0, 1, 3, 2)
        V = self.V(x).reshape(B, self.num_heads, H * W, C // self.num_heads)

        attn_score = self.attn_calculation(Q, K, V)

        attn_score = attn_score.permute(0, 1, 3, 2).reshape(B, C, H, W)
        output = self.O(attn_score)
        return output
