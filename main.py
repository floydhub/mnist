from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--dataroot', default="/input/" ,help='path to dataset')
parser.add_argument('--evalf', default="/eval/" ,help='path to evaluate sample')
parser.add_argument('--outf', default='/output',
                    help='folder to output images and model checkpoints')
parser.add_argument('--ckpf', default='',
                    help="path to model checkpoint file (to continue training)")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--train', action='store_true',
                    help='training a ConvNet model on MNIST dataset')
parser.add_argument('--evaluate', action='store_true',
                    help='evaluate a [pre]trained model')


args = parser.parse_args()
# use CUDA?
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Is there the outf?
try:
    os.makedirs(args.outf)
except OSError:
    pass

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# From MNIST to Tensor
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Load MNIST only if training
if args.train:
    train_loader = torch.utils.data.DataLoader(
        # datasets.MNIST('../data', train=True, download=True,
        datasets.MNIST(root=args.dataroot, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        # datasets.MNIST('../data', train=False, transform=transforms.Compose([
        datasets.MNIST(root=args.dataroot, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

class Net(nn.Module):
    """ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
if args.cuda:
    model.cuda()

# Load checkpoint
if args.ckpf != '':
    if args.cuda:
        model.load_state_dict(torch.load(args.ckpf))
    else:
        # Load GPU model on CPU
        model.load_state_dict(torch.load(args.ckpf, map_location=lambda storage, loc: storage))
        odel.cpu()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch):
    """Training"""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # print (data.size())
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    """Testing"""
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def test_image():
    """Take images from args.evalf, process to be MNIST compliant
    and classify them with MNIST ConvNet model"""
    def get_images_name(folder):
        """Create a generator to list images name at evaluation time"""
        onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        for f in onlyfiles:
            yield f

    def pil_loader(path):
        """Load images from /eval/ subfolder, convert to greyscale and resized it as squared"""
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                sqrWidth = np.ceil(np.sqrt(img.size[0]*img.size[1])).astype(int)
                return img.convert('L').resize((sqrWidth, sqrWidth))

    eval_loader = torch.utils.data.DataLoader(ImageFolder(root=args.evalf, transform=transforms.Compose([
                       transforms.Scale(28),
                       transforms.CenterCrop(28),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]), loader=pil_loader), batch_size=1, **kwargs)

    # Name generator
    names = get_images_name(os.path.join(args.evalf, "images"))
    model.eval()
    for data, _ in eval_loader:
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        output = model(data)
        print ("Images: ", next(names), ", Classified as: ", output.data.max(1, keepdim=True)[1])

# Train?
if args.train:
    # Train + Test per epoch
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()
    # Do checkpointing - Is saved in outf
    torch.save(model.state_dict(), '%s/mnist_convnet_model_epoch_%d.pth' % (args.outf, args.epochs))

# Evaluate?
if args.evaluate:
    test_image()
