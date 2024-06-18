import torch 
import torch.nn as nn 
from base_model import BaseModelIR
from utils.log_writer import LOGWRITER
from tqdm import tqdm
from loss import *
from utils.visualization import rgb_to_ycbcr, ycbcr_to_rgb
import os
import configs
from dataset.dataset import load_dataset

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
    
    def train_model(self, train_dl, valid_dl, optimizer, criterion, lr_scheduler, epochs, log_writer: LOGWRITER):
        """
        """
        best_loss = float('inf')
        criterion_psnr = PSNR(maximum_pixel_value=1.0)
        iteration = 0

        for epoch in range(epochs):
            self.train()
            total_tr_loss = 0.0
            for i, data in tqdm(enumerate(train_dl), total=len(train_dl)):
                optimizer.zero_grad()
                
                clean_img, degra_img = data
                
                sr_img = self(degra_img)
                
                mse_loss = criterion(clean_img, sr_img)
                mse_loss.backward()
                optimizer.step()

                total_tr_loss += mse_loss.item()
                iteration += 1

            avg_tr_loss = total_tr_loss / len(train_dl)
            avg_vld_loss, avg_vld_psnr_loss = self.evaluate_model(valid_dl, criterion, criterion_psnr)
            
            log_writer.write(epoch=epoch + 1, tr_loss=avg_tr_loss, vld_loss=avg_vld_loss, vld_psnr=avg_vld_psnr_loss)
            
            lr_scheduler.step()

            if best_loss > avg_vld_loss:
                best_loss = avg_vld_loss
                torch.save(self.state_dict(), os.path.join(configs.save_pth, f"Epoch {epoch + 1}_RESTORMER.pth"))

            train_dl, valid_dl = self.update_dataloaders_based_on_iterations(train_dl, valid_dl, iteration)

    def evaluate_model(self, valid_dl, criterion, criterion_psnr):
        """
        """
        self.eval()
        total_vld_loss = 0.0
        total_vld_psnr_loss = 0.0

        with torch.no_grad():
            for i, data in tqdm(enumerate(valid_dl)):
                clean_img, degrad_img = data
                sr_img = self(degrad_img)
                mse_loss = criterion(clean_img, sr_img)

                clean_img_YCbCr = rgb_to_ycbcr(clean_img)
                clean_img_Y = clean_img_YCbCr[:, 0, :, :]
                sr_img_YCbCr = rgb_to_ycbcr(sr_img)
                sr_img_Y = sr_img_YCbCr[:, 0, :, :]

                psnr_loss = criterion_psnr(clean_img_Y, sr_img_Y)
                total_vld_loss += mse_loss.item()
                total_vld_psnr_loss += psnr_loss.item()

        avg_vld_loss = total_vld_loss / len(valid_dl)
        avg_vld_psnr_loss = total_vld_psnr_loss / len(valid_dl)
        return avg_vld_loss, avg_vld_psnr_loss

    def update_dataloaders_based_on_iterations(self, train_dl, valid_dl, iteration):
        if iteration >= 276000:
            train_dl = load_dataset(384, 8, shuffle=True, mode="train")
            valid_dl = load_dataset(384, 8, shuffle=True, mode="val")
        elif iteration >= 240000:
            train_dl = load_dataset(320, 8, shuffle=True, mode="train")
            valid_dl = load_dataset(320, 8, shuffle=True, mode="val")
        elif iteration >= 204000:
            train_dl = load_dataset(256, 16, shuffle=True, mode="train")
            valid_dl = load_dataset(256, 16, shuffle=True, mode="val")
        elif iteration >= 156000:
            train_dl = load_dataset(192, 32, shuffle=True, mode="train")
            valid_dl = load_dataset(192, 32, shuffle=True, mode="val")
        elif iteration >= 92000:
            train_dl = load_dataset(160, 40, shuffle=True, mode="train")
            valid_dl = load_dataset(160, 40, shuffle=True, mode="val")
        return train_dl, valid_dl
        


    
