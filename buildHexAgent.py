from HexAgent import *
import multiprocessing
import os
import tensorflow as tf

def main():
    agent = HexAgent(name="bestAgent", board_size=5, batch_size=256 // multiprocessing.cpu_count(), simulations_per_state=10, max_depth=6)
    for i in range(2):
        agent.train(1)
        print (i)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    main()