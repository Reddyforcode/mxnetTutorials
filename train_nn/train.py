from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
from IPython import display
import matplotlib.pyplot as pyplot
import time

mnist_train = datasets.FashionMNIST(train=True)

X, y = mnist_train[0]
print(X)
print(y)