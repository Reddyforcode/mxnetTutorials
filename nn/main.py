import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

"""
layer = nn.Dense(2)
layer.initialize()#ctx = mx.gpu(0))
x = nd.random_uniform(-1, 1, (3, 4))
layer(x)
print(layer.weight.data)
"""

net = nn.Sequential()
net.add(
    nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Dense(120, activation='relu'),
    nn.Dense(84, activation='relu'),
    nn.Dense(10)
)

#print(net)
net.initialize()

x = nd.random.uniform(shape=(4, 1, 28, 28))
y = net(x)
print(y.shape)

