import torch 
from torch import nn
from torch.nn import functional as F  
from decoder import VAE_AttentionBlock , VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            #(batchsize, channel, Height, width) -> (batch_size, 128, heitght, width)
            nn.Conv2d(3, 128,kernel_size= 3, padding=1),

            #(batrchsize, 128, heitght, width) -> (batrchsize, 128, heitght/2, width/2)
            VAE_ResidualBlock(128, 128),

             #(batrchsize, 128, heitght, width) -> (batrchsize, 128, heitght/2, width/2)
            VAE_ResidualBlock(128, 128),
            
            #(batchsize, channel, Height, width) -> (batrchsize, 128, heitght/2, width/2)
            nn.Conv2d(128,128, kernel_size=3, padding=0, stride=2), 

            #(batrchsize, 128, heitght/2, width/2) -> (batrchsize, 256, heitght/2, width/2)
            VAE_ResidualBlock(128, 256),

            #(batrchsize, 256, heitght/2, width/2)-> (batrchsize, 256, heitght/2, width/2)
            VAE_ResidualBlock(256, 256),

            #(batrchsize, 256, heitght/2, width/2) -> (batrchsize, 256, heitght/4, width/4)
            nn.Conv2d(256,256, kernel_size=3, padding=0, stride=2), 

            #(batrchsize, 256, heitght/2, width/2) -> (batrchsize, 512, heitght/4, width/4)
            VAE_ResidualBlock(256, 512),

            #(batrchsize, 512, heitght/4, width/4)-> (batrchsize, 512, heitght/4, width/4)
            VAE_ResidualBlock(512, 512),
            
            #(batrchsize, 512, heitght/4, width/4)-> (batrchsize, 512, heitght/8, width/8)
            nn.Conv2d(512,512, kernel_size=3, padding=0, stride=2), 

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            #(batrchsize, 512, heitght/8, width/8) -> (batrchsize, 512, heitght/8, width/8)
            VAE_ResidualBlock(512, 512),

            #(batrchsize, 512, heitght/8, width/8) -> (batrchsize, 512, heitght/8, width/8)
            VAE_AttentionBlock(512, 512),

            #(batch_size, 512, heitght/8, width/8) -> (batrchsize, 512, heitght/8, width/8)
            VAE_ResidualBlock(512, 512),

            #(batrchsize, 512, heitght/8, width/8) -> (batrchsize, 512, heitght/8, width/8)
            nn.GroupNorm(32, 512),
            
            nn.SiLU(),#Activation function

            #(batrchsize, 512, heitght/8, width/8) -> (batrchsize, 8, heitght/8, width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),#BottleNeck

            #(batrchsize, 8, heitght/8, width/8) -> (batrchsize, 8, heitght/8, width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),#BottleNeck
        )

    def forward(self, x:torch.Tensor, noise:torch.Tensor)->torch.Tensor:
        #x: (batchsize, Channel, Height, Width)
        #noise: (Batch_size, Out Channel , Height/8, width/8)

        for module in self:
            if getattr(module, 'stride', None) == (2,2): 
                #Padding left, padding right, padding top, padding bottom
                x = F.pad(x, (0,1,0,1))
                x = module(x)
        # (batch_size, 8, Height, Height/8, width/8)->two tensors of shape (batch_size, 4, height/8, width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # (batch_size, 4, Height/8, width/8)->(batch_size, 4, Height/8, width/8)
        log_variance = torch.clamp(log_variance, min=-30, max=20)
        variance = log_variance.exp()
        stdev = variance.sqrt()

        #Z = N(0, 1) -> N(mean, variance)?  
        #X = mean + stdev * Z
        x = mean + stdev * noise

        # Scale the outputby a constant
        x *= 0.18215

        return x

