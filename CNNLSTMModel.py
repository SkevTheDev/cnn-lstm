import tensorflow as tf
import keras
import numpy as np
from CustomModelCheckPoint import CustomModelCheckPoint
from keras.models import Sequential 
from keras.layers import LSTM, Dense, Dropout, Flatten, TimeDistributed 
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import models
#from keras.callbacks import CSVLogger

import matplotlib.pyplot as plt

class CNNLSTMModel(object):

    def create_model(self):
        num_classes = 2
        timesteps = 20 #20 frames per sequence
        features = 50  #from hidden layer of CNN, which is the input to LSTM
        img_rows, img_cols = 64, 64
        input_shape = (timesteps, img_rows, img_cols, 3) 
        print("\ninput_shape = ", input_shape) #(20, 64, 64, 3)

        self.model = Sequential() 
        self.model.add(TimeDistributed(Conv2D(12, kernel_size=(9, 9), activation='relu', input_shape=(20, 64, 64, 3))))
        self.model.add(TimeDistributed(Conv2D(18, (7, 7), activation='relu'))) 
        self.model.add(TimeDistributed(AveragePooling2D(pool_size=(2, 2)))) 
        self.model.add(TimeDistributed(Dropout(0.25))) 
        self.model.add(TimeDistributed(Flatten())) 
        self.model.add(TimeDistributed(Dense(50, activation='relu')))
        self.model.add(TimeDistributed(Dropout(0.4)))
        self.model.add(LSTM(60, input_shape=(20, 50), return_sequences=True)) #20 timesteps, 50 features each (from CNN)
        self.model.add(Dense(num_classes, activation='sigmoid'))

        print("\nmodel.compile()") 
        self.model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(0.002), metrics=['accuracy']) 
        return self.model

    def load_model(self,model_file_name):
        print("\nloading model")
        self.model = keras.models.load_model(model_file_name)

    def train_model(self, train_images,train_labels,test_images, test_labels, epochs, batch_size):
        cbk = CustomModelCheckPoint()  # so that we can save the best model
        print("\nmodel.fit()")
        
        #csv_log = CSVLogger('epoch_valacc.csv', append=True, separator=',')

        #history = self.model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[cbk, csv_log], 
         #               validation_data=(test_images, test_labels))

        history = self.model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[cbk], 
                        validation_data=(test_images, test_labels))
        
        valacc_history = np.array(history.history['val_accuracy'])
        np.savetxt("epoch_valacc.txt", valacc_history) 

        plt.plot(history.history['val_accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy', marker='o', markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        #plt.xlim([0, 10])
        plt.ylim([0.5, 1])        
        plt.legend(loc='lower right')
        plt.show()

    def evaluate(self, test_images, test_labels):
        print("\nmodel.evaluate()") 
        test_loss, test_acc = self.model.evaluate(test_images,  test_labels, verbose=0)
        return test_loss, test_acc
