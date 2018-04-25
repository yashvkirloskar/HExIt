import numpy as np 
import scipy as sp 
import tensorflow as tf 
import HexitConvNet


#====== TESTING FILE FOR HexitConvNet CLASS ======#
batch_size = 256

#First just a sanity check to make sure shit goes through LOL
# input_shape, output_size, conv_layer_depth, batch_size, name, step_size=1
rand_inputs = np.random.rand(batch_size, 6,9,9).astype(np.float32)
rand_labels = np.random.rand(batch_size, 25).astype(np.float32)
rand_masks = np.random.rand(batch_size, 25).astype(np.float32)
convnet = HexitConvNet.CNN((6,9,9), 25, 64, batch_size, "placeholder")
convnet.train(rand_inputs, rand_labels, rand_masks, batch_size)
