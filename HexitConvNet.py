import numpy as np 
import scipy as sp 
import tensorflow as tf 

class CNN:
	def __init__(self, input_shape, output_size, conv_layer_depth, name, step_size=1, mask=None):
		self.input_shape = input_shape
		self.input_width = input_shape[0]
		self.input_height = input_shape[1]
		self.num_channels = input_shape[2]
		self.output_size = output_size
		self.conv_layer_depth = conv_layer_depth
		self.strides = [1,1]
		self.mask = mask

		#include other relevant hyperparameters in the input for init 
		self.step_size = step_size

		#output of constructing computational graph
		inputs_placeholder, labels_placeholder, ops, loss = self.buildGraph()
		self.inputs_placeholder = inputs_placeholder
		self.labels_placeholder = labels_placeholder
		self.output = ops
		self.loss = loss



	
	'''
	Building a convolutional net as described in the paper "Thinking Fast and Slow
	with Deep Learning and Tree Search"

	1) Input comes in the shape [N, N, num_channels]

	2) Layers 1-8 preserve shape so the output size from the 8th layer 9s [N, N, conv_layer_depth]

	3) Layer 9 does not pad so resultant size is [N-2, N-2, conv_layer_depth]

	4) Layer 10 does not pad so resultant size is [N-4, N-4, conv_layer_depth]

	5) Layer 11 is just 1x1 filters w/o padding so output size is [N-4, N-4, conv_layer_depth]

	6) Layer 12 preserves shape so output size is [N-4, N-4, conv_layer_depth]

	7) Layer 13 is just 1x1 filters w/o padding so output size is [N-4, N-4, conv_layer_depth]

	8) Layer 14 is a fully connected layer and output size is N-4 * N-4 * conv_layer_depth * 1/2

	9) Layer 15 is a fully connected layer and its output size is "output size"

	'''
	def buildGraph(self):

		with tf.variable_scope("convnet_computationalgraph"):

			#Setup placeholder for input
			train_inputs = tf.placeholder(tf.float32, [None, self.input_width, self.input_height, self.num_channels])
			train_labels = tf.placeholder(tf.float32, [None, self.output_size])
			output = train_inputs
	        
			#Set up convolutional layers
			for i in range(1, 14):
				output = self.buildConvolutionalOperation(output,self.conv_layer_depth, i)

			#Fully connected layers - for now I am using masks of 1 just for the sake of building this
			#TODO: change to using self.mask 

			fc_output_size_1 = np.round(self.conv_layer_depth * (self.input_height-4) * (self.input_width-4) * 1/2).astype(int).item()
			fc_mask_1 = tf.ones(tf.cast(tf.constant(fc_output_size_1), tf.int32))
			output = self.buildFullyConnectedLayer(output, fc_output_size_1, fc_mask_1)

			fc_output_size_2 = self.output_size
			fc_mask_2 = tf.ones(tf.cast(tf.constant(fc_output_size_2), tf.int32))
			output = self.buildFullyConnectedLayer(output, fc_output_size_2, fc_mask_2)

			# #Loss
			# loss = tf.reduce_mean(tf.scalar_mul(-1, tf.diag(tf.matmul(train_labels, tf.transpose(output)))))
			loss = tf.reduce_sum(output)

		return train_inputs, train_labels, output, loss




	'''
	Builds a convolutional operation to add to the computational graph based
	conv_depth = number of filters
	idx = the index of the layer 
	'''
	def buildConvolutionalOperation(self, conv_input,conv_depth,i):
		if i <= 8 or i == 12:
			output = tf.layers.conv2d(conv_input, filters=conv_depth, kernel_size=[3, 3], strides=self.strides, padding='SAME', activation=tf.nn.elu, use_bias=True)
		else:
			if i == 11 or i == 13:
				output = tf.layers.conv2d(conv_input, filters=conv_depth, kernel_size=[1, 1], strides=self.strides, padding='VALID', activation=tf.nn.elu, use_bias=True)
			else:
				output = tf.layers.conv2d(conv_input, filters=conv_depth, kernel_size=[3, 3], strides=self.strides, padding='VALID', activation=tf.nn.elu, use_bias=True)
		return tf.layers.batch_normalization(output)


	'''
	Builds a fully connected operation to add to the computational graph based
	conv_depth = number of filters
	idx = the index of the layer 
	'''
	def buildFullyConnectedLayer(self, fc_input, output_size, fc_mask):
		input_shape = tf.shape(fc_input)
		batch_size = input_shape[0]
		rest = input_shape[1] * input_shape[2] * input_shape[3]
		fc_input = tf.reshape(fc_input, [batch_size, rest])
		output = tf.layers.dense(fc_input, output_size, activation=tf.nn.softmax)
		output_masked = tf.multiply(output, fc_mask)
		return output_masked

	def train(self, inputs, labels, batch_size):
		print("training-1")
		#batch the input data
		N = len(labels)
		batched_inputs = np.split(inputs, N/batch_size)
		batched_labels = np.split(labels, N/batch_size)


		#Add an op to optimize the loss
		optimize_op = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

		print("training-2")
		# Add an op to initialize the variables.
		init_op = tf.global_variables_initializer()

		# Add ops to save and restore all the variables.
		saver = tf.train.Saver()

		print("training-3")
		with tf.Session() as session:
			init_op.run()

			print("training-4")

			average_loss = 0
			for step in range(len(batched_inputs)):

				inputs, labels = batched_inputs[step], batched_labels[step]
				feed_dict = {self.inputs_placeholder: inputs, self.labels_placeholder: labels}
				print("training-5")
				_, loss_val,output = session.run([optimize_op, self.loss, self.output], feed_dict=feed_dict)
				print(output.shape)
				average_loss += loss_val
				# print(average_loss)

				if step % 2 == 0:
					if step > 0:
						average_loss /= 2
					# The average loss is an estimate of the loss over the last 2000 batches.
					print('Average loss at step ', step, ': ', average_loss)
					average_loss = 0






