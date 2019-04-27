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


class Generator(nn.Module):

  def __init__(self, nz, nc, ngf):
    super(Generator, self).__init__()

    self.model = nn.Sequential(
        nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(ngf * 8),
        nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),
        nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Tanh()
    )

  def forward(self, x):
    return self.model(x)


class Discriminator(nn.Module):

  def __init__(self, nc, ndf):
    super(Discriminator, self).__init__()

    self.model = nn.Sequential(
        nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
        nn.Sigmoid()
    )

  def forward(self, x):
    return self.model(x)


class DCGAN():

  def __init__(self, opt):
    self.opt = opt
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    self.net_G = Generator(self.opt.nz, self.opt.nc, self.opt.ngf).to(self.device)
    self.net_D = Discriminator(self.opt.nc, self.opt.ndf).to(self.device)

    if (self.device.type == "cuda") and (self.opt.ngpu > 1):
      self.net_G = nn.DataParallel(self.net_G, list(range(self.opt.ngpu)))
      self.net_D = nn.DataParallel(self.net_D, list(range(self.opt.ngpu)))
    self.net_G.apply(weights_init)
    self.net_D.apply(weights_init)

    self.criterion = nn.BCELoss()

    self.z_fixed = torch.randn(64, self.opt.nz, 1, 1, device=self.device)

    self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
    self.optimizer_D = optim.Adam(self.net_D.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

  def set_input(self, data):
    self.real = data.to(self.device)

  def get_losses(self):
    return [self.loss_G.item(), self.loss_D.item()]

  def get_images(self):
    with torch.no_grad():
      fake_imgs = self.net_G(self.z_fixed).detach().cpu()
    return fake_imgs

  def optimize_parameters(self):
    batch_size = self.real.size(0)
    label_real = torch.full((batch_size,), 1, device=self.device)
    label_fake = torch.full((batch_size,), 0, device=self.device)

    z = torch.randn(batch_size, self.opt.nz, 1, 1, device=self.device)
    self.fake = self.net_G(z)

    pred_real = self.net_D(self.real).view(-1)
    pred_fake = self.net_D(self.fake.detach()).view(-1)

    # Update D
    self.optimizer_D.zero_grad()
    loss_D_real = self.criterion(pred_real, label_real)
    loss_D_fake = self.criterion(pred_fake, label_fake)
    self.loss_D = (loss_D_real + loss_D_fake) * 0.5
    self.loss_D.backward()
    self.optimizer_D.step()

    # Update G
    pred_fake = self.net_D(self.fake).view(-1)
    self.loss_G = self.criterion(pred_fake, label_real)
    self.optimizer_G.zero_grad()
    self.loss_G.backward()
    self.optimizer_G.step()
