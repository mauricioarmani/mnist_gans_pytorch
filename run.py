import torch
import torch.nn as nn 
from torch.autograd import Variable
from torch.optim import Adam
from torchvision.utils import save_image


def run(data, G, D, epoch):
	
	optimizer_d = Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
	optimizer_g = Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
	sig = nn.Sigmoid()
	loss = nn.BCEWithLogitsLoss()	

	for i, X in enumerate(data):
		k = 1
		x_= X[0]
		x = Variable(x_.float()).cuda()
		m = len(x)
		y_= torch.ones(m)
		y = Variable(y_.float()).cuda()
		w_= torch.zeros(m)
		w = Variable(w_.float()).cuda()

		for j in range(k):			
			z1 = torch.Tensor(m, 100).uniform_(0,1)
			z_d = Variable(z1.float()).cuda()

			loss_d_x = loss(D(x), y)

			z_d_2 = Variable(G(z_d).data, volatile=False) # Variable por causa do volatile

			loss_d_z = loss(D(z_d_2), w)

			loss_d = loss_d_x + loss_d_z

			loss_d.backward()
			optimizer_d.step()
			D.zero_grad()

		z2 = torch.Tensor(m, 100).uniform_(0,1)
		z_g = Variable(z2.float()).cuda()
		
		loss_g = loss(D(G(z_g)), y)
		
		loss_g.backward()
		optimizer_g.step()
		G.zero_grad()

		if i % 50 == 0:			
			print 'loss_d: {} - loss_g: {} - D(x): {}'.format(loss_d.data[0], loss_g.data[0], sig(D(x)).data[0])

	z3 = torch.Tensor(m, 100).uniform_(0,1)
	z_val = Variable(z3.float()).cuda()

	G.eval()
	sample = G(z_val).data
	G.train()

	sample_img = sample.div(2).add(0.5).clamp(0, 1)

	filename = 'results/results-{}.jpeg'.format(epoch+1)
	save_image(sample_img, filename, nrow=sample.shape[0]/8, normalize=True)