import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
from torch import autograd
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib.pyplot as plt
import imageio
import itertools
import numpy as np
import struct
import argparse
import seaborn as sns

from utils import *

parser = argparse.ArgumentParser()
# parser.add_argument("--num_bus", type=int, default=20, help="number of bus lines")
parser.add_argument("--code_dim", type=int, default=5, help="dim of embedded bus code")
parser.add_argument("--mu_dim", type=int, default=5, help="dim of mu")
parser.add_argument("--var_dim", type=int, default=5, help="dim of var")
parser.add_argument("--lmd", type=float, default=0.1, help="discount factor (lambda) of L1 loss in infoGAN")

parser.add_argument("--E_update_freq", type=int, default=200, help="number of epochs of training")
parser.add_argument("--save_epoch", type=int, default=200, help="number of epochs to save model")
parser.add_argument("--epoch", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00002, help="adam: learning rate")
parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--init_dim", type=int, default=100, help="dimensionality of the latent code")
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')

#parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--l2', type=float, default=0.1, help='l2 penalty')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
#parser.add_argument('--clip_value', type=float, default=20, help='gradient clipping value')

opt = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

if not os.path.exists('./train'):
    os.mkdir('./train')

if not os.path.exists('./checkpoint'):
    os.mkdir('./checkpoint')
################################ Load Data  ##########################################
########################################################################################
x = np.loadtxt(open('/home/yzhang31/bus_speed/speed_city.csv', "rb"), delimiter=",", skiprows=0)
x[x > 150] = 150
y = np.loadtxt(open('/home/yzhang31/bus_speed/buslines.csv', "rb"), delimiter=",", skiprows=0)
x = x.reshape(-1, 1, 50, 50)    # shape(1944, 1, 50, 50)
y = y.reshape(20, 1944, 50, 50)
y = y.transpose(1, 0, 2, 3)   # shape(1944, 20, 50, 50)


#time = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
#time = np.tile(time, int(x.shape[0] / 12))
# print(time.shape)  # shape(1944,)


def normalize(x):
    # min-max normalization: x -> [-1,1], because the last activation func in G is tanh
    # x is a numpy array
    max_x = x.max()
    min_x = x.min()
    x = 2 * (x - min_x) / (max_x - min_x) - 1
    return x


def de_normalize(x, max_num, min_num):
    x = x.cpu().data.numpy().reshape(-1, 1, 50, 50)
    x = 0.5 * (x + 1)
    x = x * (max_num - min_num) + min_num
    x = x[0, :, :, :]
    return x


x = x[0:1824, :, :, :]    # training data
y = y[0:1824, :, :, :]

max_num = x.max()
min_num = x.min()

x = normalize(x)
y = normalize(y)

x = torch.from_numpy(x)
y = torch.from_numpy(y)

dataset = Data.TensorDataset(x, y)
train_loader = Data.DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True)

################################ Define model ##########################################
########################################################################################
D = Discriminator(opt.code_dim)
G = Generator(opt.init_dim, opt.code_dim)
E = Embedding(opt.code_dim)
DHead = DHead()
QHead = QHead(opt.mu_dim, opt.var_dim)

if torch.cuda.device_count() > 1:
    print("number of GPU: ", torch.cuda.device_count())
    D = nn.DataParallel(D).to(device)
    G = nn.DataParallel(G).to(device)
    E = nn.DataParallel(E).to(device)
    DHead = nn.DataParallel(DHead).to(device)
    QHead = nn.DataParallel(QHead).to(device)
if torch.cuda.device_count() == 1:
    D = D.to(device)
    G = G.to(device)
    E = E.to(device)
    DHead = DHead.to(device)
    QHead = QHead.to(device)

optimD = torch.optim.Adam([{'params': D.parameters()}, {'params': DHead.parameters()}], lr=opt.lr, betas=(opt.beta1, opt.beta2))
#optimG = torch.optim.Adam([{'params': E.parameters()}, {'params': G.parameters()}, {'params': QHead.parameters()}], lr=opt.lr, betas=(opt.beta1, opt.beta2))
optimG = torch.optim.Adam([{'params': G.parameters()}, {'params': QHead.parameters()}], lr=opt.lr, betas=(opt.beta1, opt.beta2))

# define loss
cont_loss = NormalNLLLoss()

################################ Training ##############################################
########################################################################################

# List variables to store results pf training.

print("-" * 25)
print("Starting Training Loop...\n")
print("-" * 25)

real_label = 1
fake_label = 0

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []

# training
for epoch in range(opt.epoch):
    D_losses = []
    G_losses = []

    '''
    # learning rate decay
    if epoch == 10:  # or epoch == 15:
        # lr decay for D
        optimD.param_groups[0]['lr'] /= 10
        # lr decay for DHead
        optimD.param_groups[1]['lr'] /= 10
        # lr decay for E
        optimG.param_groups[0]['lr'] /= 10
        # lr decay for G
        optimG.param_groups[1]['lr'] /= 10
        # lr decay for QHead
        optimG.param_groups[2]['lr'] /= 10
    '''
    for step, (b_x, b_y) in enumerate(train_loader):
        # Get batch size
        b_size = b_x.size(0)
        # Transfer data tensor to GPU/CPU (device)
        real_x = Variable(b_x.type(dtype).to(device))
        real_y = Variable(b_y.type(dtype).to(device))

        ####### Updating discriminator and DHead ########
        for _ in range(5):
            ####### Updating discriminator and DHead ########
            optimD.zero_grad()
            # Real data
            output1 = D(real_x, real_y)
            real_code = E(real_y)
            real_logits = DHead(output1).view(-1)
            #-real_logits.mean().backward(retain_graph=True)

            # Fake data
            noise = Variable(torch.randn(b_size, opt.init_dim).type(dtype).to(device))
            fake_data = G(noise, real_code)
            output2 = D(fake_data.detach(), real_y)
            fake_logits = DHead(output2).view(-1)
            # fake_logits.mean().backward(retain_graph=True)

            # Real data with wrong label
            shuffled_idx = torch.randperm(b_size)
            real_shuffled_x = b_x[shuffled_idx]
            real_shuffled_x = Variable(real_shuffled_x.type(dtype).to(device))
            output3 = D(real_shuffled_x, real_y)
            probs = DHead(output3).view(-1)
            # probs.mean().backward(retain_graph=True)

            # gradient_penalty
            eps = torch.rand(b_size, 1, 1, 1).to(device)
            eps = eps.expand_as(real_x)
            interpolation = eps * real_x + (1 - eps) * fake_data

            interp_logits = D(interpolation, real_y)
            grad_outputs = torch.ones_like(interp_logits)

            gradients = autograd.grad(
                outputs=interp_logits,
                inputs=interpolation,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]

            gradients = gradients.view(b_size, -1)
            grad_norm = gradients.norm(2, 1)
            gradient_penalty = 10 * torch.mean((grad_norm - 1) ** 2)
            # gradient_penalty.backward(retain_graph=True)

            D_loss = fake_logits.mean() + probs.mean() - real_logits.mean() + gradient_penalty
            D_loss.backward(retain_graph=True)

            optimD.step()
            D_losses.append(D_loss.item())

        ####### Updating Generator, Embedding Net and QHead ########
        optimG.zero_grad()
        # Fake data treated as real.
        output = D(fake_data, real_y)
        # label.fill_(real_label)
        probs_fake = DHead(output).view(-1)

        gen_loss = -probs_fake.mean()

        q_mu, q_var = QHead(output)
        # Calculate loss for continuous latent code.
        code_loss = cont_loss(real_code, q_mu, q_var) * opt.lmd

        # Net loss for generator.
        G_loss = gen_loss + code_loss
        # Calculate gradients.
        G_loss.backward()
        # Update parameters.
        optimG.step()
        G_losses.append(G_loss.item())

    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

    if (epoch + 1) % opt.E_update_freq == 0:
        D_dict = D.state_dict()
        E_dict = E.state_dict()
        # 1. filter out unnecessary keys
        D_filtered_dict = {k: v for k, v in D_dict.items() if k in E_dict}
        # 2. overwrite entries in the existing state dict
        E_dict.update(D_filtered_dict)
        # 3. load the new state dict
        E.load_state_dict(E_dict)

    if (epoch + 1) % opt.save_epoch == 0:
        with torch.no_grad():
            code = E(real_y).detach()
            gen_data = G(noise, code).detach()
        fake_img = de_normalize(gen_data, max_num, min_num)
        sns.heatmap(fake_img.reshape(50, 50))  # , annot=False, vmin=0, vmax=vmax)
        plt.savefig('./train/fake_img_dim_{}_freq_{}_lmd_{}_bz_{}_initdim_{}_epoch_{}.png'.format(opt.code_dim, opt.E_update_freq, opt.lmd, opt.batch_size, opt.init_dim, epoch + 1))
        plt.close()

    # Save network weights every # epochs.
    if (epoch + 1) % opt.save_epoch == 0:
        torch.save({
            'G': G.state_dict(),
            'D': D.state_dict(),
            'DHead': DHead.state_dict(),
            'QHead': QHead.state_dict(),
            'E': E.state_dict()
        }, './checkpoint/model_dim_{}_freq_{}_lmd_{}_bz_{}_initdim_{}_epoch_{}.pth'.format(opt.code_dim, opt.E_update_freq, opt.lmd, opt.batch_size, opt.init_dim, epoch + 1))

print("-" * 25)
print('Training finished!\n')
print("-" * 25)

# Save final network weights.
torch.save({
    'G': G.state_dict(),
    'D': D.state_dict(),
    'DHead': DHead.state_dict(),
    'QHead': QHead.state_dict(),
    'E': E.state_dict()
}, './checkpoint/model_final_dim_{}_freq_{}_lmd_{}_bz_{}_initdim_{}.pth'.format(opt.code_dim, opt.E_update_freq, opt.lmd, opt.batch_size, opt.init_dim))

# Plot the training losses.
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(train_hist['G_losses'], label="G")
plt.plot(train_hist['D_losses'], label="D")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("./train/Loss_Curve_dim_{}_freq_{}_lmd_{}_bz_{}_initdim_{}.png".format(opt.code_dim, opt.E_update_freq, opt.lmd, opt.batch_size, opt.init_dim))

np.savetxt('./train/D_loss.csv', torch.FloatTensor(train_hist['D_losses']).cpu().data.numpy(), delimiter=',')
np.savetxt('./train/G_loss.csv', torch.FloatTensor(train_hist['G_losses']).cpu().data.numpy(), delimiter=',')