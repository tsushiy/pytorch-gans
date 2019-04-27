import os
import glob
import argparse

import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.utils
import torchvision.transforms as transforms


def get_options():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataroot", required=True, help="directory for dataset")
  parser.add_argument("--num_workers", type=int, default=2, help="# of subprocesses for data loading")
  parser.add_argument("--batch_size", type=int, default=32, help="batch size for training")
  parser.add_argument("--image_size", type=int, default=64, help="images will be resized to this size, this option need to be changed with networks")
  parser.add_argument("--mode", type=str, default="train", help="mode: train or test")
  parser.add_argument("--input_nc", type=int, default=3, help="# of input channels")
  parser.add_argument("--output_nc", type=int, default=3, help="# of output channels")
  parser.add_argument("--ngf", type=int, default=64, help="# of filters in generator")
  parser.add_argument("--ndf", type=int, default=64, help="# of filters in discriminator")
  parser.add_argument("--load_epoch", type=int, default=0, help="which epoch to load. 0 for not to load")
  parser.add_argument("--num_epochs", type=int, default=2, help="# of training epochs without decaying lr")
  parser.add_argument("--num_epochs_decay", type=int, default=2, help="# of training epochs with decaying lr")
  parser.add_argument("--save_epoch_freq", type=int, default=1, help="how often to save models")
  parser.add_argument("--visualize_iter_freq", type=int, default=4, help="how often to visualize images")
  parser.add_argument("--lr", type=float, default=0.0002, help="learning rate for optimizer")
  parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam")
  parser.add_argument("--lambda_cyc", type=float, default=10.0, help="weight for cycle loss")
  parser.add_argument("--lambda_idt", type=float, default=5.0, help="weight for identity loss")
  parser.add_argument("--ngpu", type=int, default=1, help="# of gpus. 0 for cpu mode")
  opt = parser.parse_args()
  return opt


def plot_losses(losses):
  plt.figure(figsize=(10, 5))
  plt.title("Losses")
  loss_names = ["G_A", "G_B", "D_A", "D_B", "idt_A", "idt_B"]
  for i, name in enumerate(loss_names):
    plt.plot(losses[i], label=name)
  plt.xlabel("Iteration")
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig("./results/losses.png")
  plt.close()


def save_images(imgs, epoch, iters):
  torchvision.utils.save_image(imgs, "./results/epoch%d_iter%d.png" % (epoch, iters), normalize=True)


def create_dataloader(opt):

  dataset = CustomDataset(
      root=opt.dataroot,
      mode=opt.mode,
      transform=transforms.Compose([
          transforms.Resize(opt.image_size),
          transforms.CenterCrop(opt.image_size),
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      ]))

  dataloader = torch.utils.data.DataLoader(
      dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
  return dataloader


class CustomDataset(torch.utils.data.Dataset):

  def __init__(self, root, mode="train", transform=None):
    self.loader = torchvision.datasets.folder.default_loader
    self.transform = transform

    self.filenames_A = sorted(glob.glob(os.path.join(root, "%sA" % mode, "*.jpg")))
    self.filenames_B = sorted(glob.glob(os.path.join(root, "%sB" % mode, "*.jpg")))

  def __getitem__(self, index):
    image_A = self.loader(self.filenames_A[index % len(self.filenames_A)])
    image_B = self.loader(self.filenames_B[index % len(self.filenames_B)])
    if self.transform is not None:
      image_A = self.transform(image_A)
      image_B = self.transform(image_B)

    return image_A, image_B

  def __len__(self):
    return max(len(self.filenames_A), len(self.filenames_B))
