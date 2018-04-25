import numpy as np 
import scipy as sp 
import tensorflow as tf 
import HexitConvNet


#====== TESTING FILE FOR HexitConvNet CLASS ======#

#First just a sanity check to make sure shit goes through LOL

rand_inputs = np.random.rand(1000, 9,9,6).astype(np.float32)
rand_labels = np.random.rand(1000, 2).astype(np.float32)
convnet = HexitConvNet.CNN((9,9,6), 2, 5, "placeholder")
convnet.train(rand_inputs, rand_labels, 2)
