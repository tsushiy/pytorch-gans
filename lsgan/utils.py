import argparse

import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms


def get_options():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataroot", required=True, help="directory for dataset")
  parser.add_argument("--num_workers", type=int, default=2, help="# of subprocesses for data loading")
  parser.add_argument("--batch_size", type=int, default=128, help="batch size for training")
  parser.add_argument("--image_size", type=int, default=64, help="images will be resized to this size, this option need to be changed with networks")
  parser.add_argument("--nc", type=int, default=3, help="# of channels")
  parser.add_argument("--nz", type=int, default=100, help="size of latent space")
  parser.add_argument("--ngf", type=int, default=64, help="# of filters in generator")
  parser.add_argument("--ndf", type=int, default=64, help="# of filters in discriminator")
  parser.add_argument("--num_epochs", type=int, default=5, help="# of training epochs")
  parser.add_argument("--lr", type=float, default=0.0002, help="learning rate for optimizer")
  parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam")
  parser.add_argument("--ngpu", type=int, default=1, help="# of gpus. 0 for cpu mode")
  opt = parser.parse_args()
  return opt


def create_dataloader(opt):
  dataset = torchvision.datasets.ImageFolder(
      root=opt.dataroot,
      transform=transforms.Compose([
          transforms.Resize(opt.image_size),
          transforms.CenterCrop(opt.image_size),
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      ]))

  dataloader = torch.utils.data.DataLoader(
      dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
  return dataloader


def plot_losses(G_losses, D_losses):
  plt.figure(figsize=(10, 5))
  plt.title("Losses")
  plt.plot(G_losses, label="G")
  plt.plot(D_losses, label="D")
  plt.xlabel("Iteration")
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig("./results/losses.png")
  plt.close()


def save_images(imgs, epoch, iters):
  torchvision.utils.save_image(imgs, "./results/epoch%d_iter%d.png" % (epoch, iters), normalize=True)
