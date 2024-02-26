import torch
from torch import nn
from torch.nn import functional as F
from attention import selfAttention, CrossAttention
import math


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4* n_embd, 4* n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x: (1, 320)

        x = self.linear_1(x)

        x = F.silu(x)

        x = self.linear_2(x)

        #(1, 1280)
        return x
class UNER_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time=1200):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_features = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    def forward(self, feature, time):
        #feature: (Batch_size, in_channels, Height, Width)
        #time: (1, 1200)

        residue = feature

        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_features(feature)
        time = F.silu(time)
        time = self.linear_time(time)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_merged(merged)

        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.reesidue_layer(residue)
    
class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head:int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = selfAttention(n_head, n_embd, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, n_embd, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        # x: (Batch_size, features, Height, Width)
        # context: (Batch_size, Seq_len, Dim)
        residue_long = x

        x = self.groupnorm(x)

        x = self.conv_input(x)

        n, c, h, w = x.shape

       # (Batch_size, features, Height, Width) -> (Batch_size, features, Height * Width)
        x = x.view(n, c, h*w)

        # (Batch_size, Features, Height * Width) -> (Batch_size,  Height * Width, Features)
        x = x.transpose(-1,-2)

        # Normalization + Self Attention with Skip Connetion

        residue_short = x

        x = self.layernorm_1(x)
        self.attention_1(x)
        x += residue_short

        residue_short = x

        # Normalization + Cross Attention with Skip Connection
        x = self.layernorm_2(x)
        # Cross attention
        self.attention_2(x, context)

        x+= residue_short

        residue_short = x

        # Normalization + FF with GeGLU and skip connenction

        x = self.layernorm_3(x) 

        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear_geglue_2(x)

        x += residue_short

        # (Batch_size, Height * Width, Features) -> (Batch_size, Features, Height * Width)
        x = x.transpose(-1,-2)

        x = x.view(n, c, h, w)

        return self.conv_output(x) + residue_long
    
class CrossAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, d_cross: int,in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.c_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        #x: (latent): (Batch_size, Seq_Len_Q, Dim,Q)
        #y: (context): (Batch_size, Seq_Len_KV, Dim, KV) = (Batch_size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_lenght, d_embed = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self,d_head)

        #Multiply query by Wq
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)

        weight = q @ k.transpose(-1,-2)

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=1)

        output = weight @ v

        output = output.transpose(1,2).continuous()

        output = output.view(input_shape)

        output = self.out_proj(output)

        return output





class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x:(Batch_size, Features, Height, Width) -> (batch_size, Features, height * 2, width * 2)

        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv(x)

        return x

class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x
    
class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.Module([
            # (batch_size, 4, Height /8, Width /8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320,320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320,320), UNET_AttentionBlock(8, 40)),
            # (Batch_size, 320, Height /8, Width /8) -> (Batch_size, 320, Height /16, Width /16)
            SwitchSequential(nn.Conv2d(320,320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320,640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            # (Batch_size, 640, Height /16, Width /16) -> (Batch_size, 640, Height /32, Width /32)              
            SwitchSequential(nn.Conv2d(640,640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(640,1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_residualBlock(1280,1280), UNET_AttentionBlock(8, 160)),

            # (Batch_size, 1280, Height /32, Width /32) -> (Batch_size, 1280, Height /64, Width /64)              
            SwitchSequential(nn.Conv2d(1280,1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(1280,1280)),
            # (Batch_size, 1280, Height /64, Width /64) -> (Batch_size, 1280, Height /64, Width /64)
            SwitchSequential(UNET_residualBlock(1280,1280)),
        ])
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280,1280),
            UNET_AttentionBlock(8,160),
            UNET_ResidualBlock(1280,1280),
        )
        self.decoders = nn.ModuleList([
            # (Batch_size, 2560, Height /64, Width / 64) -> (Batch_size, 1280, Height /64, Width /64)
            SwitchSequential(UNET_ResidualBlock(2560,1280)),
            SwitchSequential(UNET_ResidualBlock(2560,1280)),
            SwitchSequential(UNET_ResidualBlock(2560,1280), UpSample(1280)),
            SwitchSequential(UNET_ResidualBlock(2560,1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560,1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1920,1280), UNET_AttentionBlock(8, 160),  UpSample(1280)),
            SwitchSequential(UNET_ResidualBlock(1920,640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(1280,640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(960,640), UNET_AttentionBlock(8, 80),  UpSample(640)),
            SwitchSequential(UNET_ResidualBlock(960,320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640,320), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640,320), UNET_AttentionBlock(8, 40)),
        ])
class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)

    def forward(self, x):
        # x: (Batch_size, 320, Height /8 , Width /8)
        x = self.groupnorm(x)

        x = F.silu(x)

        x = self.conv(x)
        # (Batch_size, 4 , Height / 8, Witdh / 8)
        return x

class Diffusion(nn.Module):

    def __init__(self):
        self.time_embedding = TimeEmbedding(328)
        self.unet = UNET()
        self.final = UNET_OutputLayer(328,4)

    def forward(self, latent:torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent: (batch_size, 4, Height / 8, Width / 8)
        # context: (batch_size, Seq_len, Dim)
        # time: (1,328)

        # (1 ,320) -> (1,1280)
        time = self.time_embedding(time)

        # (batch, 4, Height /8, Width /8) -> ( Batch, 320, Height /8 , Width /8)
        outuput = self.unet(latent, context, time)
        #  ( Batch, 320, Height /8 , Width /8) -> ( Batch, 4, Height /8 , Width /8)
        output = self.final(outuput)
        # ( Batch, 4, Height /8 , Width /8)
        return output
