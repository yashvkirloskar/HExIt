import numpy as np 
import scipy as sp 
import tensorflow as tf 

class ConvNet:
    def __init__(input_shape, conv_layer_depth, mean, sd):
        # Input shape should be of size [batch_size, width, height, num_channels]
        self.input_width = input_shape[0]
        self.input_height = input_shape[1]
        self.strides = [1, 1]
        self.conv_layer_depth = conv_layer_depth
        self.filters = []
        for i in range(13):
            if i == 0:
                self.filters.append(tf.Variable(tf.random_normal(shape=[3, 3, input_shape[2]], mean, sd)))
            elif i == 10 and i == 12:
                # Filters for layers 11 and 13 are [1, 1, num_channels]
                self.filters.append(tf.Variable(tf.random_normal(shape=[1, 1, conv_layer_depth, conv_layer_depth], mean, sd)))
            else:
                self.filters.append(tf.Variable(tf.random_normal(shape=[3, 3, conv_layer_depth, conv_layer_depth], mean, sd)))
                
        self.filter_biases = []
        for i in range(13):
            self.filter_biases.append(tf.Variable(tf.zeros([conv_layer_depth])))

        self.scales = []
        self.offsets = []
        self.means = []
        self.variances = []
        self.var_eps = 1e-12
        for i in range(13):
            self.means.append(tf.Variable(tf.zeros(shape=[conv_layer_depth])))
            self.variances.append(tf.Variable(tf.zeros(shape=[conv_layer_depth])))
            self.scales.append(tf.Variable(tf.ones(shape=[conv_layer_depth])))
            self.offsets.append(tf.Variable(tf.zeros(shape=[conv_layer_depth])))


    def train(inputs, labels, mode, mask):
        outputs = [inputs]
        # Do convolution with ELU non-linearity and a batch-norm
        # Stride is always 1
        for i in range(1, 14):
            if i <= 8 or i == 12:
                # Convolve with padding
                output = tf.layers.conv2d(input=outputs[i-1], filters=self.conv_layer_depth, kernel_size=[3, 3], strides=strides, padding='SAME', activation=tf.nn.elu, use_bias=True)   
            else:
                # Convolve without padding
                if i == 11 or i == 13:
                    output = tf.layers.conv2d(input=outputs[i-1], filters=self.conv_layer_depth, kernel_size=[1, 1], strides=strides, padding='VALID', activation=tf.nn.elu, use_bias=True)
                else:
                    output = tf.layers.conv2d(input=outputs[i-1], filters=self.conv_layer_depth, kernel_size=[3, 3], strides=strides, padding='VALID', activation=tf.nn.elu, use_bias=True)
                
            output = tf.nn.batch_normalization(output, self.means[i-1], self.variances[i-1], self.offsets[i-1], self.scales[i-1], self.var_eps)
            outputs.append(output)

        # Apply the mask to the last layer of the output
        masked = tf.multiply(outputs[-1], mask)

        # Need to apply two fully parallel softmax layers


        if mode == 'test':
            return outputs[-1]


        # Need to apply the loss too

