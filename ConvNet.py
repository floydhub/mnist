from __future__ import print_function
import argparse
import numpy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable

class Net(nn.Module):
    """ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ConvNet(object):
    """MNIST ConvNet Model class"""
    def __init__(self,
                 ckp="/input/mnist_convnet_model_epoch_10.pth",
                 evalf="/eval"):
        """MNIST ConvNet Builder

        Args:
            ckp: path to model checkpoint file (to continue training).
            evalf: path to evaluate sample.
        """
        # Path to model weight
        self._ckp = ckp
        # Use CUDA?
        self._cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if self._cuda else "cpu")
        try:
            os.path.isfile(ckp)
            self.ckp = ckp
        except IOError as e:
            # Does not exist OR no read permissions
            print ("Unable to open ckp file")
        self._evalf = evalf


    # Build the model loading the weights
    def build_model(self):
        self._model = Net().to(self._device)

        # Load Weights
        if self._cuda:
            self._model.load_state_dict(torch.load(self._ckp))
        else:
            # Load GPU model on CPU
            self._model.load_state_dict(torch.load(self._ckp, map_location=lambda storage, loc: storage))


    # Preprocess Images to be MNIST-compliant
    def image_preprocessing(self):
        """Take images from args.evalf, process to be MNIST compliant
        and classify them with MNIST ConvNet model"""
        def pil_loader(path):
            """Load images from /eval/ subfolder, convert to greyscale and resized it as squared"""
            with open(path, 'rb') as f:
                with Image.open(f) as img:
                    sqrWidth = np.ceil(np.sqrt(img.size[0]*img.size[1])).astype(int)
                    return img.convert('L').resize((sqrWidth, sqrWidth))

        kwargs = {'num_workers': 1, 'pin_memory': True} if self._cuda else {}
        self._eval_loader = torch.utils.data.DataLoader(ImageFolder(root=self._evalf,
            transform=transforms.Compose([
                           transforms.Resize(28),
                           transforms.CenterCrop(28),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]), loader=pil_loader), batch_size=1, **kwargs)

    def classify(self):
        """Classify the current eval batch"""
        self._model.eval()
        with torch.no_grad():
            for data, target in self._eval_loader:
                data = data.to(self._device)
                output = self._model(data)
                label = output.argmax(dim=1, keepdim=True).item()
                return label
