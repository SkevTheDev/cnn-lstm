#Ranjana Koshy
#CPSE 710
#Face Liveness Detection using CNN-LSTM, on video frames (Replay-Attack dataset)

#Ref:https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
#https://machinelearningmastery.com/cnn-long-short-term-memory-networks/


from __future__ import print_function
import tensorflow 
import keras 
from ReadData import ReadData
from CustomModelCheckPoint import CustomModelCheckPoint
from CNNLSTMModel import CNNLSTMModel
import tensorflow as tf
import os
import sys
import numpy as np
import time

import matplotlib.pyplot as plt

def main():
    readd = ReadData()

    train_images, train_labels = readd.read_data_train()
    val_images, val_labels = readd.read_data_val()

    #test_images, test_labels, test_images_real, test_labels_real, test_images_attack, test_labels_attack = readd.read_data_test()

    cnnlstm = CNNLSTMModel()   
    start_time = time.time()    

    cnnlstm.create_model()  # create and train a new model
    cnnlstm.train_model(train_images, train_labels, val_images, val_labels, 100, 32) #100: epochs, 32: batch size
    #cnnlstm.train_model(train_images, train_labels, test_images, test_labels, 100, 32) #100: epochs, 32: batch size
    
    #cnnlstm.load_model('model.h5') #to use pretrained model
    #cnnlstm.train_model(train_images,train_labels,val_images,val_labels, 100, 32)
    #cnnlstm.train_model(train_images,train_labels,test_images,test_labels, 100, 32)
    
    print("after training model")
    end_time = time.time()   
    time_elapsed = end_time - start_time    
    minutes = int(time_elapsed/60)
    seconds = time_elapsed - (minutes * 60)
    print("\n" + "time taken (training) = " + str(minutes) + "minutes " + str(round(seconds, 2)) + "seconds")
    
    start_time1 = time.time()    

    test_loss, test_acc = cnnlstm.evaluate(val_images, val_labels)
    #test_loss, test_acc = cnnlstm.evaluate(test_images, test_labels)
    print('Test loss:', test_loss) 
    print('Test accuracy:', test_acc)

    end_time1 = time.time()   
    time_elapsed = end_time1 - start_time1    
    minutes = int(time_elapsed/60)
    seconds = time_elapsed - (minutes * 60)
    print("\n" + "time taken (evaluation) = " + str(minutes) + "minutes " + str(round(seconds, 2)) + "seconds")

    ############Computing HTER##################
    """
    print("\n" + "Computing HTER:")

    print("\n" + "model.evaluate() -> real test images") 
    score_real = cnnlstm.evaluate(test_images_real, test_labels_real) 
    print('Test loss (real):', score_real[0]) 
    print('Test accuracy (real):', score_real[1])

    print("\n" + "model.evaluate() -> attack test images") 
    score_attack = cnnlstm.evaluate(test_images_attack, test_labels_attack) 
    print('Test loss (attack):', score_attack[0]) 
    print('Test accuracy (attack):', score_attack[1])

    FRR = 100 - (score_real[1] * 100)
    FAR = 100 - (score_attack[1] * 100)

    print("\nFRR = " + str(FRR) + ", FAR = " + str(FAR))

    hter = (FRR + FAR)/2

    print("\nhter = " + str(hter))
    """
if __name__ == "__main__":
    sys.exit(int(main() or 0))