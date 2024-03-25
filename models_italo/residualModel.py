import torch
import torch.nn as nn
#from einops.layers.torch import Reduce
import sys


#class Repeat(Reduce):
#    def __init__(self, pattern, **axes_lengths):
#        super().__init__(pattern, 'repeat', **axes_lengths)

class residualCon(nn.Module):
    def __init__(self, encoder, decoder, compose=lambda x,y: torch.sum([x,y], dim=1)):
        super().__init__()
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)
        self.pool =  nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.compose = compose
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
    def save(self, path):
        torch.save(self.state_dict(), path)

    def forward(self, input):

        for enc in self.encoder:
                residual = input
                input = enc(input)
                if input.shape[2] != residual.shape[2]:
                     downSampler = nn.Conv2d(residual.shape[1], input.shape[1], kernel_size=1, stride=2, bias=False).to(device=input.device)
                else:
                    downSampler = nn.Conv2d(residual.shape[1], input.shape[1], kernel_size=1, stride=1, bias=False).to(device=input.device)
                downSampled = downSampler(residual)
                input += downSampled
        

        for dec in self.decoder:
            input = dec(input)
        return input
