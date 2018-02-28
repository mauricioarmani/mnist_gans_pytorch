from models import Discriminator, Generator
from run import run
from loader import load_data
from torchvision.datasets import MNIST
import torch


BATCH_SIZE = 128
train_data = load_data(BATCH_SIZE)

Z_DIM = 100
G = Generator(Z_DIM).cuda()
D = Discriminator().cuda()

z_val_ = torch.Tensor(BATCH_SIZE, 100).normal_(0,1)

epochs = 100
for epoch in range(epochs):
	print 'epoch: %d/%d' % (epoch+1, epochs)
	run(train_data, G, D, epoch, z_val_)