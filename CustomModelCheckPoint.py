import tensorflow as tf
import keras
from keras.callbacks import Callback

class CustomModelCheckPoint(keras.callbacks.Callback):
    def __init__(self):
        self.lastvalacc = 0 # last best

    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
        
        print("epoch: {epoch}, train_acc: {logs['accuracy']}, valid_acc: {logs['val_accuracy']}")
        
        #if logs['val_accuracy'] > logs['accuracy']: # your custom condition
        if logs['val_accuracy'] > self.lastvalacc: # your custom condition
            print("\nsaving better model")
            self.model.save('model.h5', overwrite=True)
            self.lastvalacc = logs['val_accuracy']
            print("self.lastvalacc = ", self.lastvalacc)
            print('best model on epoch found')            
