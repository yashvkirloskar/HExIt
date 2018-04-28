import numpy as np 
import scipy as sp 
import tensorflow as tf 
import os

class CNN:
    def __init__(self, input_shape, output_size, conv_layer_depth, batch_size, name,  step_size=1):
        self.input_shape = input_shape
        self.num_channels = input_shape[0]
        self.input_width = input_shape[1]
        self.input_height = input_shape[2]

        self.batch_size = batch_size
        self.output_size = output_size
        self.conv_layer_depth = conv_layer_depth
        self.strides = [1,1]

        #include other relevant hyperparameters in the input for init 
        self.step_size = step_size

        #name used for saving and restoring
        self.name = name

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

        # with tf.variable_scope("convnet_computationalgraph"):
        #Setup placeholder for input
        train_inputs = tf.placeholder(tf.float32, [None, self.num_channels, self.input_width, self.input_height], name="inputs")
        train_labels = tf.placeholder(tf.float32, [None, self.output_size], name="labels")
        mask = tf.placeholder(tf.float32, [None, self.output_size], name="mask")

        # Put tensor in correct shape (NHWC)
        output = tf.transpose(train_inputs, [0, 2, 3, 1])

        #Set up convolutional layers
        for i in range(1, 14):
            output = self.buildConvolutionalOperation(output,self.conv_layer_depth, i)

        # Put inputs in terms of player 1 and player 2
        outputWhite, outputBlack = tf.split(output, 2, axis=0)
        maskWhite, maskBlack = tf.split(mask, 2, axis=0)
        labelsWhite, labelsBlack = tf.split(train_labels, 2, axis=0)

        # Using a single FC layer loss
        outputP1 = self.buildFullyConnectedLayerWithSoftmax(outputWhite, self.output_size, maskWhite)
        outputP2 = self.buildFullyConnectedLayerWithSoftmax(outputBlack, self.output_size, maskBlack)

        # #Loss
        lossP1 = tf.scalar_mul(-1, tf.reduce_mean(tf.reduce_sum(tf.multiply(labelsWhite, tf.log1p(outputP1)), axis=1)))
        lossP2 = tf.scalar_mul(-1, tf.reduce_mean(tf.reduce_sum(tf.multiply(labelsBlack, tf.log1p(outputP2)), axis=1)))

        output = tf.stack((outputP1, outputP2), axis=0, name="output")
        loss = tf.stack((lossP1, lossP2), axis=0, name="loss")

        return train_inputs, train_labels, mask, output, loss

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
    def buildFullyConnectedLayerWithSoftmax(self, fc_input, output_size, fc_mask):
        # input_shape = tf.shape(fc_input)
        # 5x5x64
        rest = (self.input_width-4) * (self.input_height-4) * self.conv_layer_depth
        fc_input = tf.reshape(fc_input, [-1, rest])

        output = tf.layers.dense(fc_input, output_size)
        masked = tf.multiply(output, fc_mask)
        output = tf.nn.softmax(masked, axis=1)
        temp = tf.multiply(output, fc_mask)
        output = tf.divide(temp, tf.reduce_sum(tf.add(temp, 1e-12), axis=1, keepdims=True))
        return output

    def train(self, inputs, labels, mask):


        if(os.path.exists(self.name + "/convnet.meta")):
            print ("Loading from previous checkpoint")
            # Add an op to initialize the variables.
            init_op = tf.global_variables_initializer()
        
            with tf.Session() as session:

                #Restore the old graph and get relevant ops / variables
                saver = tf.train.import_meta_graph(self.name + "/" + 'convnet.meta')
                saver.restore(session,tf.train.latest_checkpoint(self.name))
                graph = tf.get_default_graph()
                input_op = graph.get_tensor_by_name("inputs:0")
                mask_op = graph.get_tensor_by_name("mask:0")
                output_op = graph.get_tensor_by_name("output:0")
                labels_op = graph.get_tensor_by_name("labels:0")
                loss_op = graph.get_tensor_by_name("loss:0")
                optimize_op = tf.get_collection('optimizer')[0]
                feed_dict = {input_op:inputs, mask_op:mask, labels_op:labels}

                #initialize variables
                init_op.run()

                _,loss_val, output = session.run([optimize_op, loss_op, output_op], feed_dict=feed_dict)
                total_loss = loss_val
                save_path = saver.save(session, self.name +  "/" + "convnet")
                print ("===========Result Ops=============")
                print (len(output))
                print (output[0].shape)
                print (output[0][0])
                print ("===========Result Loss=============")
                print (len(loss_val))
                print (loss_val[0])

        else:
            print("no saved graph, building a new one")

            new_graph = tf.Graph()
            with tf.Session(graph=new_graph) as session:

                inputs_placeholder, labels_placeholder, mask_placeholder, ops, loss = self.buildGraph()

                # Add an op to initialize the variables.
                init_op = tf.global_variables_initializer()

                #Add an op to optimize the loss
                optimize_op = tf.train.GradientDescentOptimizer(self.step_size).minimize(loss)


                # Add ops to save and restore all the variables.
                saver = tf.train.Saver()

                tf.add_to_collection("optimizer", optimize_op)

                init_op.run()
                feed_dict = {inputs_placeholder: inputs, labels_placeholder: labels, mask_placeholder: mask}
                _,result_ops, result_loss = session.run([optimize_op,  ops, loss] , feed_dict=feed_dict)
                save_path = saver.save(session, self.name +  "/" + "convnet")
                print ("===========Result Ops=============")
                print (len(result_ops))
                print (result_ops[0].shape)
                print (result_ops[0][0])
                print ("===========Result Loss=============")
                print (len(result_loss))
                print (result_loss[0])

            



    def predict(self, predict_input, predict_mask):
        # Add an op to initialize the variables.
    
        with tf.Session() as session:

            #Restore the old graph and get relevant ops / variables
            saver = tf.train.import_meta_graph(self.name + "/" + 'convnet.meta')
            saver.restore(session,tf.train.latest_checkpoint(self.name))
            graph = tf.get_default_graph()
            inputs = graph.get_tensor_by_name("inputs:0")
            mask = graph.get_tensor_by_name("mask:0")
            feed_dict = {inputs: predict_input, mask: predict_mask}
            op_to_restore = graph.get_tensor_by_name("output:0")
            output = session.run([op_to_restore], feed_dict=feed_dict)

        return output

