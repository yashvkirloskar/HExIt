from Expert import *
from Apprentice import *
import os
import shutil
import multiprocessing
import tensorflow as tf


def testBasic():
    if(os.path.exists("testExpert")):
        shutil.rmtree("testExpert")
    apprentice = Apprentice("testExpert", 5, 4)
    expert = Expert(board_size=5, batch_size=4, simulations_per_state=10, max_depth=4)
    batch, labels = expert.generateBatch()
    apprentice.train(batch, labels)
    expert = Expert(board_size=5, batch_size=4, simulations_per_state=10, max_depth=4, apprentice=apprentice)
    batch, labels = expert.generateBatch()
    apprentice.train(batch, labels)
    print ("Test complete")
    if(os.path.exists("testExpert")):
        shutil.rmtree("testExpert")

def main():
    print ("Starting tests")
    testBasic()

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
    multiprocessing.set_start_method('spawn')
    main()