import torch 
from torch import nn 
from torch import functional as F 
from sd.attention import selfAttention 


class VAE_AttentionBlock(nn.Module):

    def __init__(self, channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = selfAttention(1, channels)
    def forward(self,x: torch.Tensor)->torch.Tensor:
        #x = (batch_size, features, height, width)
        residue = x

        n, c, h, w = x.shape
        #x = (batch_size, features, height, width) -> (batch_size,features, height * width  )
        x = x.view(n, c, h*w)

        #x = (batch_size,features, height * width  ) -> (batch_size, height* width, features)
        x = x.transpose(-1,-2)

        #(batch_size, height * width,features  ) -> (batch_size, height* width, features)
        x = self.attention(x)

        #x = (batch_size,, height * width ,features ) -> (batch_size,features, height* width)
        x = x.transpose(-1,-2)

        #(batch_size,features, height* width) -> (batch_size, features, height, width)
        x =x.view(n, c, h, w)

        x += residue

        return x


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = (batch_size, in_channels, height, width)

        residue = x

        h = self.groupnorm_1(F.silu(self.conv1(x)))
        h = self.groupnorm_2(F.silu(self.conv2(h)))
        return h + self.residual_layer(residue)

class VAE_Decoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(4,4, kernel_size=1, padding=0),

            nn.Conv2d(4,512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            #(Batch_Size, 512, height / 8, width / 8) -> (Batch_Size, 512, height / 8, width / 8)
            VAE_ResidualBlock(512, 512),

            #(Batch_Size, 512, height / 8, width / 8) -> (batch_size, Height/4, width / 4)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size = 3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            #(Batch_Size, 512, height / 4, width / 4) -> (batch_size, 512, Height / 2, width / 2)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size = 3, padding=1),

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            #(Batch_Size, 256, height / 2, width / 2) -> (batch_size,256, Height , width )
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size = 3, padding=1),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),

            nn.SiLU(),

            #(batch_size, 128, Height, Width) -> (batch_size, 3, Height, Width) 
            nn.Conv2d(128, 3, kernel_size = 3, padding = 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #(batch_size, 4, height / 8, width /8)

        x /= 0.18215

        for module in self:
            x = module(x)

        # (batch_size, 3, Height, width)
        return x