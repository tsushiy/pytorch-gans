import os

import models
import utils


if __name__ == "__main__":
  opt = utils.get_options()
  model = models.DCGAN(opt)
  dataloader = utils.create_dataloader(opt)

  os.makedirs("./results", exist_ok=True)

  G_losses, D_losses = [], []

  print("Start Training...")

  for epoch in range(1, opt.num_epochs + 1):
    for iters, (data, _) in enumerate(dataloader):
      model.set_input(data)
      model.optimize_parameters()

      loss_G, loss_D = model.get_losses()
      G_losses.append(loss_G)
      D_losses.append(loss_D)

      if iters % 50 == 0:
        print("Epoch: %d/%d\tIter: %d/%d\tLoss_G: %.4f\tLoss_D: %.4f" %
              (epoch, opt.num_epochs, iters, len(dataloader), loss_G, loss_D))

      if (iters % 500 == 0) or ((epoch == opt.num_epochs) and (iters == len(dataloader) - 1)):
        utils.save_images(model.get_images(), epoch, iters)
        utils.plot_losses(G_losses, D_losses)
