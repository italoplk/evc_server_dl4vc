import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn

class NNmodel(nn.Module):

    def __init__(self, no_skip, num_filters):
        super(NNmodel, self).__init__()

        n_filters = 32

        # MAINTAIN THE SAME ENCODER 1 CHANNEL 32X32
        # concatenating the last layer 1536x2x2
        self.cnn1 = nn.Sequential(
                nn.Conv2d(1, n_filters, 3, stride=2, padding=1), nn.PReLU(),  # 32²
               
            
                nn.Conv2d(n_filters, (n_filters * 2), 3, stride=2, padding=1), nn.PReLU(),  # 16²
                

                nn.Conv2d((n_filters*2), (n_filters*4), 3, stride=2, padding=1), nn.PReLU(),  # 8²
                

                nn.Conv2d((n_filters*4), 512, 3, stride=2, padding=1), nn.PReLU(),  # 4²  
        )

        # Setting up the Fully Connected Layers
        self.decoder = nn.Sequential(  # 10, 4
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 8
                nn.Conv2d(1536, n_filters*4, kernel_size=3, stride=1, padding=1), nn.PReLU(),

                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 8
                nn.Conv2d((n_filters*4), n_filters * 2, kernel_size=3, stride=1, padding=1), nn.PReLU(),
           
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 8
                nn.Conv2d( (n_filters * 2), n_filters, kernel_size=3, stride=1, padding=1), nn.PReLU(),
          

                #MOVE TO A NEW FILE
                nn.ConvTranspose2d((n_filters), 1, 4, stride=2, padding=1),
                
                #no conv trans to get rid of checker pattern on first block
                #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 8
                #nn.Conv2d(mul_fact *(n_filters), 1, kernel_size=3, stride=1, padding=1), nn.PReLU(),
                nn.Sigmoid()
           
        )
        
    
    def forward(self, input1, input2, input3):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.cnn1(input1)
        output2 = self.cnn1(input2)
        output3 = self.cnn1(input3)
        
        out = torch.cat((output1, output2, output3), 1)
        output = self.decoder(out)

        return output

#from argparse import Namespace
#params = Namespace()
#dims = (8,1,64,64)
#dims_out = (8,1,32,32)
#(params.num_views_ver, params.num_views_hor, params.predictor_size, params.predictor_size) = dims
#params.num_filters = 16
#params.skip = False
## print(params)
#model = SiameseNetwork(params)
#model.eval()
##zeros = torch.zeros(1, 1, 64, 64)
##zeros_t = torch.zeros(8, 1, 32, 32)
##lossf = nn.MSELoss()
###
#from torchsummary import summary
#with torch.no_grad():
###     batch_size = model(zeros)
###     # print("batch_size: ", batch_size.shape)
###     # batch_size = batch_size[:,:,-32:, -32:]
###
#   summary(model, [(1, 32, 32),(1, 32, 32),(1, 32, 32)])
    # print(batch_size.shape)
    #batch_size = batch_size[:, :, -32:, -32:]
    # print(batch_size.shape)

    # print(lossf(zeros_t, batch_size))