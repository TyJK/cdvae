import torch

model = torch.load("/home/jake/Projects/mila/cdvae/assets/checkpoints/mp-plain-specific-epoch=59-step=54659.ckpt")

# a hack to change full colab-specific paths in ckpts to generic repo paths
model_hparams_copy = model["hyper_parameters"].copy()
model_hparams_copy = str(model_hparams_copy)
model_hparams_copy = model_hparams_copy.replace("/content/cdvae/", "./")
model_hparams_copy = eval(model_hparams_copy)
model["hyper_parameters"] = model_hparams_copy

torch.save(model, "/home/jake/Projects/mila/cdvae/assets/checkpoints/mp-plain-pi/epoch=59-step=54659.ckpt")
