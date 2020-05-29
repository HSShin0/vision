import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """A simple Convolution Block.

    Conv2d - Activation - MaxPool2d - Conv2d - Activation - MaxPool2d
    """
    def __init__(self, activation=F.relu_):
        pass


class ConvNet(nn.Module):
    """A simple Convolution Neural Network for classification."""
    def __init__(self):
       super().__init__()
       self.conv_module = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1),  # [8, 28, 28]
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(2),  # [8, 14, 14]
                                        nn.Conv2d(8, 16, 3, padding=1),  # [16, 14, 14]
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(2)  # [16, 7, 7]
                                        )
       out_dim = 16*7*7
       self.fc_module = nn.Sequential(nn.Linear(out_dim, 64),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(64, 32),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(32, 10)
                                      )
       self._initialize_params()

    def _initialize_params(self):
        """Initialize training parameters."""
        for m in self.children():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if hasattr(m, 'bias'):
                    nn.init.zeros_(m.bias)
        print("Initialized training parameters.")

    def forward(self, x):
        """ x.shape = [batch_size, 1, 28, 28] for MNIST """
        x = self.conv_module(x)
        x = self.flatten(x) 
        score = self.fc_module(x)
        return score

    def flatten(self, x):
        """Flatten a tensor x.""" 
        dims = x.size()
        dim = 1
        for d in dims[1: ]:
            dim *= d
        return x.view(-1, dim)

    def save(self, fp='backup.pth'):
        """Save the model to the filepath ./save/{fp}"""
        if not os.path.exists('./save'):
            os.makedirs('./save')
        filename = os.path.join('./save', fp)
        torch.save({'state_dict': self.state_dict()})
        print(f"Saved the state dict of the model to {filename}")

    def load(self, fp):
        """Load the model from the filepath ./save/{fp}"""
        filename = os.path.join('./save', fp)
        assert os.path.exists(filename), 'wrong filename'
        
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint['state_dict'])
