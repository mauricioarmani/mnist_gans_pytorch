from models import Discriminator, Generator
from run import run
from loader import load_data
from torchvision.datasets import MNIST


batch_size = 128
train_data = load_data(batch_size)

z_dim = 100
G = Generator(z_dim).cuda()
D = Discriminator().cuda()

epochs = 25
for epoch in range(epochs):
	print 'epoch: %d/%d' % (epoch+1, epochs)
	run(train_data, G, D, epoch)