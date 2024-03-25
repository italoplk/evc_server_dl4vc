import torch
import torch.nn as nn
#from einops.layers.torch import Reduce
import sys

#MODEL WITH NO SKIP

class RegModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)

        
    def load(self, path):
        self.load_state_dict(torch.load(path))
    def save(self, path):
        torch.save(self.state_dict(), path)

    def forward(self, in1):
        for i,enc in enumerate(self.encoder):
            #print(f'enc {i}: {in1.shape}')
            in1 = enc(in1)
            #print(f'output {in1.shape}\n')
        # print('-----')
        # print("enc2", X.shape)
            #out = torch.cat((in1, in2, in3), dim=1)

        for i, (dec) in enumerate(self.decoder):
            #print(f'dec {i}: {in1.shape}')
            in1 = dec(in1)
            #print(f'output {in1.shape}\n')
        # print(X)

        #     print("shape", s)
        # X = self.decoder[-1](X)
        return in1
