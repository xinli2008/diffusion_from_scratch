import torch

image_size = 48
timestep = 1000
max_epoch = 200
batch_size = 1024
device = "cuda:3" if torch.cuda.is_available() else "cpu"