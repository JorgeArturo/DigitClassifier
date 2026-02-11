import torch
from fastai.vision.all import *
import matplotlib.pyplot as plt


#add MNIST Dataset
path = untar_data(URLs.MNIST_SAMPLE)

valid_3_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'3').ls()])
valid_7_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens.float()/255
valid_3_tens = valid_3_tens.float()/255
print(valid_3_tens.shape, valid_7_tens.shape)

def mnist_distance(a,b):
    return (a-b).abs().mean((-1,-2))

mnist_distance(a_3,mean_3)
