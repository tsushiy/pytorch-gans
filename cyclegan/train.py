import os

import models
import utils


if __name__ == "__main__":
  opt = utils.get_options()
  model = models.CycleGAN(opt)
  if opt.load_epoch > 0:
    model.load_networks(opt.load_epoch)
  dataloader = utils.create_dataloader(opt)

  os.makedirs("./results", exist_ok=True)
  os.makedirs("./checkpoints", exist_ok=True)

  losses = [[] for _ in range(6)]

  print("Start Training...")

  for epoch in range(1, opt.num_epochs + opt.num_epochs_decay + 1):
    for iters, data in enumerate(dataloader):
      model.set_input(data)
      model.optimize_parameters()

      for idx, loss in enumerate(model.get_losses()):
        losses[idx].append(loss)

      if iters % opt.visualize_iter_freq == 0:
        print("Epoch: %d/%d Iter: %d Loss_G_A: %.4f Loss_G_B: %.4f Loss_D_A: %.4f Loss_D_B: %.4f Loss_idt_A: %.4f Loss_idt_B: %.4f" %
              (epoch, opt.num_epochs + opt.num_epochs_decay, iters, losses[0][-1], losses[1][-1], losses[2][-1], losses[3][-1], losses[4][-1], losses[5][-1]))

      if (iters % opt.visualize_iter_freq == 0) or ((epoch == opt.num_epochs) and (iters == len(dataloader) - 1)):
        utils.save_images(model.get_images(), epoch, iters)
        utils.plot_losses(losses)

    if epoch % opt.save_epoch_freq == 0:
      model.save_networks(epoch)
      print("Saved the model at epoch %d" % epoch)

    model.update_learning_rate()
