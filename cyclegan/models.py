import os
import itertools

import torch
import torch.nn as nn
import torch.optim as optim


def weights_init(m):
  classname = m.__class__.__name__
  if classname.find("Conv") != -1:
    nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find("BatchNorm") != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)


class ResBlock(nn.Module):
  def __init__(self, dim):
    super(ResBlock, self).__init__()

    self.block = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim, dim, kernel_size=3, padding=0),
        nn.BatchNorm2d(dim),
        nn.ReLU(True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim, dim, kernel_size=3, padding=0),
        nn.BatchNorm2d(dim)
    )

  def forward(self, x):
    out = x + self.block(x)
    return out


class Generator(nn.Module):

  def __init__(self, input_nc, output_nc, ngf, n_res_blocks=9):
    super(Generator, self).__init__()

    model = []

    model += [
        nn.ReflectionPad2d(3),
        nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True)
    ]

    mult = 1

    for _ in range(2):
      model += [
          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
          nn.BatchNorm2d(ngf * mult * 2),
          nn.ReLU(True)
      ]
      mult *= 2

    for _ in range(n_res_blocks):
      model += [ResBlock(ngf * mult)]

    for _ in range(2):
      model += [
          nn.ConvTranspose2d(ngf * mult,
                             (ngf * mult) // 2,
                             kernel_size=3,
                             stride=2,
                             padding=1,
                             output_padding=1),
          nn.BatchNorm2d((ngf * mult) // 2),
          nn.ReLU(True)
      ]
      mult //= 2

    model += [
        nn.ReflectionPad2d(3),
        nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
        nn.Tanh()
    ]

    self.model = nn.Sequential(*model)

  def forward(self, x):
    return self.model(x)


class Discriminator(nn.Module):

  def __init__(self, input_nc, ndf):
    super(Discriminator, self).__init__()

    model = []

    model += [
        nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, True)
    ]

    mult = 1

    for _ in range(2):
      model += [
          nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(ndf * mult * 2),
          nn.LeakyReLU(0.2, True)
      ]
      mult *= 2

    model += [
        nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=1),
        nn.BatchNorm2d(ndf * mult * 2),
        nn.LeakyReLU(0.2, True)
    ]
    mult *= 2

    model += [nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=1)]

    self.model = nn.Sequential(*model)

  def forward(self, x):
    return self.model(x)


class GANLoss(nn.Module):

  def __init__(self):
    super(GANLoss, self).__init__()
    self.register_buffer('label_real', torch.tensor(1.0))
    self.register_buffer('label_fake', torch.tensor(0.0))
    self.loss = nn.MSELoss()

  def forward(self, input, target):
    if target == 1:
      loss = self.loss(input, self.label_real.expand_as(input))
    else:
      loss = self.loss(input, self.label_fake.expand_as(input))
    return loss.mean()


class CycleGAN():

  def __init__(self, opt):
    self.opt = opt
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    self.net_G_A = Generator(self.opt.input_nc, self.opt.output_nc, self.opt.ngf).to(self.device)
    self.net_G_B = Generator(self.opt.input_nc, self.opt.output_nc, self.opt.ngf).to(self.device)
    self.net_D_A = Discriminator(self.opt.input_nc, self.opt.ndf).to(self.device)
    self.net_D_B = Discriminator(self.opt.input_nc, self.opt.ndf).to(self.device)

    self.network_names = ["G_A", "G_B", "D_A", "D_B"]

    if (self.device.type == "cuda") and (self.opt.ngpu > 1):
      self.net_G_A = nn.DataParallel(self.net_G_A, list(range(self.opt.ngpu)))
      self.net_G_B = nn.DataParallel(self.net_G_B, list(range(self.opt.ngpu)))
      self.net_D_A = nn.DataParallel(self.net_D_A, list(range(self.opt.ngpu)))
      self.net_D_B = nn.DataParallel(self.net_D_B, list(range(self.opt.ngpu)))
    self.net_G_A.apply(weights_init)
    self.net_G_B.apply(weights_init)
    self.net_D_A.apply(weights_init)
    self.net_D_B.apply(weights_init)

    self.criterion_GAN = GANLoss().to(self.device)
    self.criterion_cycle = nn.L1Loss()
    self.criterion_idt = nn.L1Loss()

    self.optimizer_G = optim.Adam(itertools.chain(self.net_G_A.parameters(), self.net_G_B.parameters()), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
    self.optimizer_D = optim.Adam(itertools.chain(self.net_D_A.parameters(), self.net_D_B.parameters()), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    def lr_lambda(epoch):
      lr = 1.0 - max(0, epoch - opt.num_epochs_decay) / opt.num_epochs_decay
      return lr

    self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lr_lambda)
    self.scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=lr_lambda)

  def set_input(self, data):
    self.real_A = data[0].to(self.device)
    self.real_B = data[1].to(self.device)

  def get_losses(self):
    return [self.loss_G_A.item(), self.loss_G_B.item(), self.loss_D_A.item(), self.loss_D_B.item(), self.loss_idt_A.item(), self.loss_idt_B.item()]

  def get_images(self):
    images = torch.cat([
        self.real_A.narrow(0, 0, 1),
        self.fake_B.narrow(0, 0, 1),
        self.rec_A.narrow(0, 0, 1),
        self.real_B.narrow(0, 0, 1),
        self.fake_A.narrow(0, 0, 1),
        self.rec_B.narrow(0, 0, 1)],
        dim=0)
    return images

  def update_learning_rate(self):
    self.scheduler_G.step()
    self.scheduler_D.step()
    print("learning rate = %.7f" % self.optimizer_G.param_groups[0]["lr"])

  def save_networks(self, epoch):
    for name in self.network_names:
      net = getattr(self, "net_" + name + "")
      path = os.path.join("./checkpoints", "epoch%d_%s" % (epoch, name) + ".pth")
      torch.save(net.state_dict(), path)

  def load_networks(self, epoch):
    for name in self.network_names:
      net = getattr(self, "net_" + name)
      path = os.path.join("./checkpoints", "epoch%d_%s" % (epoch, name) + ".pth")
      net.load_state_dict(torch.load(path))
      net.to(self.device)

  def forward(self):
    self.fake_B = self.net_G_A(self.real_A)
    self.rec_A = self.net_G_B(self.fake_B)
    self.fake_A = self.net_G_B(self.real_B)
    self.rec_B = self.net_G_A(self.fake_A)

  def backward_D(self):
    pred_real_A = self.net_D_A(self.real_A).view(-1)
    pred_fake_A = self.net_D_A(self.fake_A.detach()).view(-1)
    pred_real_B = self.net_D_B(self.real_B).view(-1)
    pred_fake_B = self.net_D_B(self.fake_B.detach()).view(-1)

    loss_D_A_real = self.criterion_GAN(pred_real_A, 1)
    loss_D_A_fake = self.criterion_GAN(pred_fake_A, 0)
    self.loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
    self.loss_D_A.backward()

    loss_D_B_real = self.criterion_GAN(pred_real_B, 1)
    loss_D_B_fake = self.criterion_GAN(pred_fake_B, 0)
    self.loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
    self.loss_D_B.backward()

  def backward_G(self):
    pred_fake_A = self.net_D_A(self.fake_A.detach()).view(-1)
    pred_fake_B = self.net_D_B(self.fake_B.detach()).view(-1)

    self.loss_G_A = self.criterion_GAN(pred_fake_A, 1)
    self.loss_G_B = self.criterion_GAN(pred_fake_B, 1)

    self.loss_cycle_A = self.criterion_cycle(self.rec_A, self.real_A) * self.opt.lambda_cyc
    self.loss_cycle_B = self.criterion_cycle(self.rec_B, self.real_B) * self.opt.lambda_cyc

    if self.opt.lambda_idt > 0:
      self.loss_idt_A = self.criterion_idt(self.net_G_A(self.real_B), self.real_B) * self.opt.lambda_idt
      self.loss_idt_B = self.criterion_idt(self.net_G_B(self.real_A), self.real_A) * self.opt.lambda_idt
    else:
      self.loss_idt_A = 0
      self.loss_idt_B = 0

    self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
    self.loss_G.backward()

  def optimize_parameters(self):
    self.forward()

    self.optimizer_G.zero_grad()
    self.backward_G()
    self.optimizer_G.step()

    self.optimizer_D.zero_grad()
    self.backward_D()
    self.optimizer_D.step()
