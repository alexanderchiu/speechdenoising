from __future__ import division
import neurolab as nl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

fs, ss = wavfile.read('./data/car_lom.wav')  # sampled observed signal
ss[:10]
L = len(ss)

z = np.zeros(9)

ss_pad = np.hstack((z,ss))

inp = np.zeros((L,10))
for i in range(L):
	inp[i,:] = ss_pad[i:i+10][::-1]

ss = ss.reshape((L,1))
ds = SupervisedDataSet(10, 1)

ds.setField('input', inp)
ds.setField('target', ss)



nn = FeedForwardNetwork()
inLayer = LinearLayer(10)
hiddenLayer = TanhLayer(4)
outLayer = LinearLayer(1)

nn.addInputModule(inLayer)
nn.addModule(hiddenLayer)
nn.addOutputModule(outLayer)

in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

nn.addConnection(in_to_hidden)
nn.addConnection(hidden_to_out)

nn.sortModules()

trainer = BackpropTrainer(nn, ds)

print trainer.trainUntilConvergence()
