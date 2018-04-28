import numpy as np 
import scipy as sp 
import tensorflow as tf 
import HexitConvNet
import copy
import shutil


#====== TESTING FILE FOR HexitConvNet CLASS ======#
batch_size = 256


def convert_to_softmax(labels):
    return labels / np.sum(labels, axis=1, keepdims=True)




###########  Basic Sanity Check ###########
rand_inputs = np.random.rand(2* batch_size, 6,9,9).astype(np.float32)
rand_labels = convert_to_softmax(np.random.rand(2* batch_size, 25).astype(np.float32))
rand_masks = np.ones((2* batch_size, 25)).astype(np.float32)
convnet = HexitConvNet.CNN((6,9,9), 25, 64, batch_size, "test_dir1")
convnet.train(rand_inputs, rand_labels, rand_masks)

rand_inputs2 = np.random.rand(2* batch_size, 6,9,9).astype(np.float32)

output1 = convnet.predict(rand_inputs2, rand_masks)
output2 = convnet.predict(copy.deepcopy(rand_inputs2),rand_masks)

print ("==========PRINT RANDIINPUTS============")
print (rand_inputs[0, 0, 0, 0])
print ("=============PRINT RANDLABELS============")
print (rand_labels[0, 0])
print ("============PRINT RANDMASK===============")
print (rand_masks[0, 0])
print("===========PRINT OUTPUT1===========")
print(output1)
print (len(output1))
print(output1[0].shape)
print (output1[0][0, 0])
print("============PRINT OUTPUT2=============")
print(output1)
print (len(output1))
print(output1[0].shape)
print (output1[0][0, 0])

###########################################



###########   Testing Basic Save and Restore capabilities  ###########
print("Testing to see if weights are saved after training: SHOULD BE TRUE")
print(np.array_equal(output1, output2))


convnet2 = HexitConvNet.CNN((6,9,9), 25, 64, batch_size, "test_dir1")
output3 = convnet2.predict(copy.deepcopy(rand_inputs2), copy.deepcopy(rand_masks))
print("Testing to see if weights are reinitialized in a new CNN with same name: SHOULD BE TRUE")
print(np.array_equal(output3, output2))
print ("===============PRINT OUTPUT3=================")
print (len(output3))
print (output3[0].shape)
print (output3[0][0, 0])
#####################################################################



###########   Testing to see if the CNN starts training from the original data ###########
rand_inputs3 = np.random.rand(2* batch_size, 6,9,9).astype(np.float32)
rand_labels3 = convert_to_softmax(np.random.rand(2* batch_size, 25).astype(np.float32))
convnet.train(rand_inputs3, rand_labels3, rand_masks)

output4 = convnet.predict(copy.deepcopy(rand_inputs2),rand_masks)
print ("===========OUTPUT4=============")
print (len(output4))
print (output4[0].shape)
print (output4[0][0, 0])

convnet3 = HexitConvNet.CNN((6,9,9), 25, 64, batch_size, "test_dir2")
convnet3.train(rand_inputs3, rand_labels3, rand_masks)

output5 = convnet3.predict(copy.deepcopy(rand_inputs2),rand_masks)

print("Testing to see if the neural net trains starting from its last saved spot: SHOULD BE FALSE")
print(np.array_equal(output4, output5))
output4 = np.array(output4)
output5 = np.array(output5)
print(np.sum((output5-output4)**2))
##########################################################################################

# Test with some zero masks
rand_inputs4 = np.random.rand(2*batch_size, 6, 9, 9).astype(np.float32)
rand_masks_with_zero1 = np.random.choice([0, 1], [2*batch_size, 25]).astype(np.float32)
rand_labels4 = convert_to_softmax(np.random.rand(2* batch_size, 25).astype(np.float32)  * rand_masks_with_zero1)
convnet3.train(rand_inputs4, rand_labels4, rand_masks_with_zero1)

rand_inputs5 = np.random.rand(2*batch_size, 6, 9, 9).astype(np.float32)
rand_masks_with_zero2 = np.random.choice([0, 1], [2*batch_size, 25]).astype(np.float32)
output6 = convnet3.predict(rand_inputs5, rand_masks_with_zero2)
print ("========OUTPUT6=========")
print (len(output6))
print (output6[0].shape)
print (rand_masks_with_zero2[0])
print (output6[0][0, 0])
print (np.sum(output6[0][0, 0]))
assert(np.count_nonzero(output6) == np.count_nonzero(rand_masks_with_zero2))

# Test with some zero masks
rand_inputs6 = np.random.rand(2*batch_size, 6, 9, 9).astype(np.float32)
rand_masks_with_zero3 = np.random.choice([0, 1], [2*batch_size, 25]).astype(np.float32)
rand_labels6 = convert_to_softmax(np.random.rand(2* batch_size, 25).astype(np.float32)  * rand_masks_with_zero1)
convnet4 = HexitConvNet.CNN((6, 9, 9), 25, 64, batch_size, "test_dir3")
convnet4.train(rand_inputs6, rand_labels6, rand_masks_with_zero3)

rand_inputs7 = np.random.rand(2*batch_size, 6, 9, 9).astype(np.float32)
rand_masks_with_zero4 = np.random.choice([0, 1], [2*batch_size, 25]).astype(np.float32)
output7 = convnet4.predict(rand_inputs7, rand_masks_with_zero4)
print ("========OUTPUT7=========")
print (len(output7))
print (output7[0].shape)
print (rand_masks_with_zero4[0])
print (output7[0][0, 0])
print (np.sum(output7[0][0, 0]))
assert(np.count_nonzero(output7) == np.count_nonzero(rand_masks_with_zero4))

rand_input_8 = np.random.rand(2, 6, 9, 9).astype(np.float32)
rand_input_8[1] = np.ones((6, 9, 9)).astype(np.float32)
rand_mask_with_zero5 = np.random.choice([0, 1], [2, 25]).astype(np.float32)
rand_mask_with_zero5[1] = np.zeros(25)
output8 = convnet4.predict(rand_input_8, rand_mask_with_zero5)
print ("==========OUTPUT8============")
print (len(output8))
print (output8[0].shape)
print (rand_mask_with_zero5[0])
print (rand_mask_with_zero5[1])
print (output8[0][0])
print (output8[0][1])
assert(np.count_nonzero(output8) == np.count_nonzero(rand_mask_with_zero5))

#Delete all directories at end
shutil.rmtree("test_dir1")
shutil.rmtree("test_dir2") 
shutil.rmtree("test_dir3") 
