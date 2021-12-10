from PIL import Image
import numpy as np
import keras 
from keras import backend as K 
import os

class ReadData(object):

    def read_data_train(self):
        # input image dimensions 
        img_rows, img_cols = 64, 64
        num_classes = 2 

        #Reading the train images
        tr_Images = []
        tr_Labels = []
        tr_Filenames = []

        trainImagesDirectory = "/home/skev/FaceLivenessDetection_CNNLSTM/ReplayAttackdataset/DiffusedImages/Matlab15/frames20TrainingDiffused/"        
        print(trainImagesDirectory)
        trainImages = os.listdir(trainImagesDirectory)
        
        trainIndexFile = "/home/skev/FaceLivenessDetection_CNNLSTM/ReplayAttackdataset/DiffusedImages/frames20TrainingDiffused.txt";

        for trainImage in trainImages:    
            fobj = open(trainIndexFile,"r")
            for line in fobj:            
                filename, label = line.split("\t") #the filename and label are read from the .txt file            
                lbl = int(label); #int object                
                if(trainImage == filename):
                    fname = os.path.join(trainImagesDirectory, trainImage);                                  
                    break;
            fobj.close();   
            #print("fname = ", fname)
            #print("lbl = ", lbl)
            image = Image.open(fname, mode = 'r').convert('RGB') #reading an image. convert('RGB') is needed when original images are read, not needed for diffused images
            image = np.array(image) #the 2-d array of integer pixel values    
            image = image/255.0     #the 2-d array of float pixel values (between 0 and 1)
            tr_Images.append(image) #adding the train image to list                        
            tr_Labels.append(lbl)   #adding the train label to list
            tr_Filenames.append(fname); #adding the train image filename to list
    
        n = len(trainImages)
        print("\n" + "number of training images = " + str(n)); #7200
        print("length of tr_Images = " + str(len(tr_Images))); #7200
        print("length of tr_Labels = " + str(len(tr_Labels))); #7200

        tr_Images =np.reshape(np.array(tr_Images), (360, 20, 64, 64, 3)) #360 sequences of 20 frames each
        tr_Labels = np.reshape(np.array(tr_Labels), (360, 20))
        tr_Filenames = np.reshape(np.array(tr_Filenames), (360, 20))

        print("\n..tr_Images.shape = ", tr_Images.shape) #(360, 20, 64, 64, 3)
        print("..tr_Labels.shape = ", tr_Labels.shape) #(360, 20)
        print("..tr_Filenames.shape = ", tr_Filenames.shape) #(360, 20)

        (X_train, Y_train) = tr_Images, tr_Labels

        if K.image_data_format() == 'channels_first': 
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 3, img_rows, img_cols)         
        else:     
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], img_rows, img_cols, 3)         
            
        # convert class vectors to binary class matrices 
        Y_train = keras.utils.to_categorical(Y_train, num_classes) 

        print("\n" + "after keras.utils.to_categorical()") 

        print ("number of training examples = " + str(X_train.shape[0])) #360
        print ("X_train shape: " + str(X_train.shape)) #(360, 20, 64, 64, 3)
        print ("Y_train shape: " + str(Y_train.shape)) #(360, 20, 2)

        return X_train, Y_train
        
    def read_data_val(self):
        # input image dimensions 
        img_rows, img_cols = 64, 64
        num_classes = 2 

        #Reading the validation images
        val_Images = []
        val_Labels = []
        val_Filenames = []
    
        valImagesDirectory = "/home/skev/FaceLivenessDetection_CNNLSTM/ReplayAttackdataset/DiffusedImages/Matlab15/frames20DevelopmentDiffused/"
        print("\n" + valImagesDirectory)
        valImages = os.listdir(valImagesDirectory)

        valIndexFile = "/home/skev/FaceLivenessDetection_CNNLSTM/ReplayAttackdataset/DiffusedImages/frames20DevelopmentDiffused.txt";

        for valImage in valImages:    
            fobj = open(valIndexFile,"r")
            for line in fobj:            
                filename, label = line.split("\t") #the filename and label are read from the .txt file            
                lbl = int(label); #int object                
                if(valImage == filename):
                    fname = os.path.join(valImagesDirectory, valImage);                                  
                    break;
            fobj.close();   
            #print("fname = ", fname)
            #print("lbl = ", lbl)
            image = Image.open(fname, mode = 'r').convert('RGB') #reading an image
            image = np.array(image) #the 2-d array of integer pixel values    
            image = image/255.0     #the 2-d array of float pixel values (between 0 and 1)
            val_Images.append(image) #adding the val image to list                        
            val_Labels.append(lbl)   #adding the val label to list
            val_Filenames.append(fname); #adding the val image filename to list
    
        n = len(valImages)
        print("\n" + "number of validation images = " + str(n)); #7200
        print("length of val_Images = " + str(len(val_Images))); #7200
        print("length of val_Labels = " + str(len(val_Labels))); #7200

        val_Images = np.reshape(np.array(val_Images), (360, 20, 64, 64, 3))
        val_Labels = np.reshape(np.array(val_Labels), (360, 20))
        val_Filenames = np.reshape(np.array(val_Filenames), (360, 20))

        print("\n..val_Images.shape = ", val_Images.shape) #(360, 20, 64, 64, 3)
        print("..val_Labels.shape = ", val_Labels.shape) #(360, 20)
        print("..val_Filenames.shape = ", val_Filenames.shape) #(360, 20)

        (X_val, Y_val) = val_Images, val_Labels

        if K.image_data_format() == 'channels_first': 
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 3, img_rows, img_cols) 
        else:     
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], img_rows, img_cols, 3)         

        # convert class vectors to binary class matrices 
        Y_val = keras.utils.to_categorical(Y_val, num_classes) 

        print("\n" + "after keras.utils.to_categorical()") 

        print ("number of validation examples = " + str(X_val.shape[0])) #360
        print ("X_val shape: " + str(X_val.shape))     #(360, 20, 64, 64, 3)
        print ("Y_val shape: " + str(Y_val.shape))     #(360, 20, 2)   

        return X_val, Y_val

    def read_data_test(self):
        # input image dimensions 
        img_rows, img_cols = 64, 64
        num_classes = 2 

        #Reading the test images
        test_Images = []
        test_Labels = []
        test_Filenames = []

        testImagesDirectory = "/home/skev/FaceLivenessDetection_CNNLSTM/ReplayAttackdataset/DiffusedImages/Matlab15/frames20TestingDiffused/"
        print("\n" + testImagesDirectory)
        testImages = os.listdir(testImagesDirectory)

        testIndexFile = "/home/skev/FaceLivenessDetection_CNNLSTM/ReplayAttackdataset/DiffusedImages/frames20TestingDiffused.txt";

        for testImage in testImages:    
            fobj = open(testIndexFile,"r")
            for line in fobj:            
                filename, label = line.split("\t") #the filename and label are read from the .txt file            
                lbl = int(label); #int object                
                if(testImage == filename):
                    fname = os.path.join(testImagesDirectory, testImage);                                  
                    break;
            fobj.close();   
            #print("fname = ", fname)
            #print("lbl = ", lbl)            
            image = Image.open(fname, mode = 'r').convert('RGB') #reading an image.
            image = np.array(image)   #the 2-d array of integer pixel values    
            image = image/255.0       #the 2-d array of float pixel values (between 0 and 1)    
            test_Images.append(image) #adding the test image to list                        
            test_Labels.append(lbl)   #adding the test label to list 
            test_Filenames.append(fname) #adding the test image filename to list 

        n = len(testImages)
        print("\n" + "number of test images = " + str(n)); #9600
        print("length of test_Images = " + str(len(test_Images))); #9600
        print("length of test_Labels = " + str(len(test_Labels))); #9600

        test_Images = np.reshape(np.array(test_Images), (480, 20, 64, 64, 3)) #480 sequences of 20 frames each
        test_Labels = np.reshape(np.array(test_Labels), (480, 20))
        test_Filenames = np.reshape(np.array(test_Filenames), (480, 20))

        print("\n..test_Images.shape = ", test_Images.shape) #(480, 20, 64, 64, 3)
        print("..test_Labels.shape = ", test_Labels.shape) #(480, 20)
        print("..test_Filenames.shape = ", test_Filenames.shape) #(480, 20)

        (X_test, Y_test) = test_Images, test_Labels

        ###########For HTER measurement##################

        #Reading the real test images
        test_Images_real = []
        test_Labels_real = []
        test_Filenames_real = []

        testImagesRealDirectory = "/home/skev/FaceLivenessDetection_CNNLSTM/ReplayAttackdataset/DiffusedImages/Matlab15/frames20TestingDiffused_real/"
        print("\n" + testImagesRealDirectory)
        testImages_real = os.listdir(testImagesRealDirectory)

        testIndexFile_real = "/home/skev/FaceLivenessDetection_CNNLSTM/ReplayAttackdataset/DiffusedImages/frames20TestingDiffused_real.txt";

        for testImage_real in testImages_real:    
            fobj = open(testIndexFile_real,"r")
            for line in fobj:            
                filename, label = line.split("\t") #the filename and label are read from the .txt file            
                lbl = int(label); #int object                
                if(testImage_real == filename):
                    fname = os.path.join(testImagesRealDirectory, testImage_real);                                  
                    break;
            fobj.close();   
            #print("fname = ", fname)
            #print("lbl = ", lbl)            
            image = Image.open(fname, mode = 'r').convert('RGB') #reading an image.
            image = np.array(image)   #the 2-d array of integer pixel values    
            image = image/255.0       #the 2-d array of float pixel values (between 0 and 1)    
            test_Images_real.append(image) #adding the real test image to list                        
            test_Labels_real.append(lbl)   #adding the real test label to list 
            test_Filenames_real.append(fname) #adding the real test image filename to list 

        n = len(testImages_real)
        print("\n" + "number of test images = " + str(n)); #1600
        print("length of test_Images_real = " + str(len(test_Images_real))); #1600
        print("length of test_Labels_real = " + str(len(test_Labels_real))); #1600

        test_Images_real = np.reshape(np.array(test_Images_real), (80, 20, 64, 64, 3)) #80 sequences of 20 frames each
        test_Labels_real = np.reshape(np.array(test_Labels_real), (80, 20))
        test_Filenames_real = np.reshape(np.array(test_Filenames_real), (80, 20))

        print("\n..test_Images_real.shape = ", test_Images_real.shape) #(80, 20, 64, 64, 3)
        print("..test_Labels_real.shape = ", test_Labels_real.shape) #(80, 20)
        print("..test_Filenames_real.shape = ", test_Filenames_real.shape) #(80, 20)

        (X_test_real, Y_test_real) = test_Images_real, test_Labels_real

        #Reading the attack test images
        test_Images_attack = []
        test_Labels_attack = []
        test_Filenames_attack = []

        testImagesAttackDirectory = "/home/skev/FaceLivenessDetection_CNNLSTM/ReplayAttackdataset\DiffusedImages/Matlab15/frames20TestingDiffused_attack/"
        print("\n" + testImagesAttackDirectory)
        testImages_attack = os.listdir(testImagesAttackDirectory)

        testIndexFile_attack = "/home/skev/FaceLivenessDetection_CNNLSTM/ReplayAttackdataset/DiffusedImages/frames20TestingDiffused_attack.txt";

        for testImage_attack in testImages_attack:    
            fobj = open(testIndexFile_attack,"r")
            for line in fobj:            
                filename, label = line.split("\t") #the filename and label are read from the .txt file            
                lbl = int(label); #int object                
                if(testImage_attack == filename):
                    fname = os.path.join(testImagesAttackDirectory, testImage_attack);                                  
                    break;
            fobj.close();   
            #print("fname = ", fname)
            #print("lbl = ", lbl)            
            image = Image.open(fname, mode = 'r').convert('RGB') #reading an image.
            image = np.array(image)   #the 2-d array of integer pixel values    
            image = image/255.0       #the 2-d array of float pixel values (between 0 and 1)    
            test_Images_attack.append(image) #adding the attack test image to list                        
            test_Labels_attack.append(lbl)   #adding the attack test label to list 
            test_Filenames_attack.append(fname) #adding the attack test image filename to list 

        n = len(testImages_attack)
        print("\n" + "number of test images = " + str(n)); #8000
        print("length of test_Images_attack = " + str(len(test_Images_attack))); #8000
        print("length of test_Labels_attack = " + str(len(test_Labels_attack))); #8000

        test_Images_attack = np.reshape(np.array(test_Images_attack), (400, 20, 64, 64, 3)) #400 sequences of 20 frames each
        test_Labels_attack = np.reshape(np.array(test_Labels_attack), (400, 20))
        test_Filenames_attack = np.reshape(np.array(test_Filenames_attack), (400, 20))

        print("\n..test_Images_attack.shape = ", test_Images_attack.shape) #(400, 20, 64, 64, 3)
        print("..test_Labels_attack.shape = ", test_Labels_attack.shape) #(400, 20)
        print("..test_Filenames_attack.shape = ", test_Filenames_attack.shape) #(400, 20)

        (X_test_attack, Y_test_attack) = test_Images_attack, test_Labels_attack

        if K.image_data_format() == 'channels_first': 
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 3, img_rows, img_cols) 
            X_test_real = X_test_real.reshape(X_test_real.shape[0], X_test_real.shape[1], 3, img_rows, img_cols)     
            X_test_attack = X_test_attack.reshape(X_test_attack.shape[0], X_test_attack.shape[1], 3, img_rows, img_cols)     
        else:     
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], img_rows, img_cols, 3)    
            X_test_real = X_test_real.reshape(X_test_real.shape[0], X_test_real.shape[1], img_rows, img_cols, 3)     
            X_test_attack = X_test_attack.reshape(X_test_attack.shape[0], X_test_attack.shape[1], img_rows, img_cols, 3)     

        # convert class vectors to binary class matrices 
        Y_test = keras.utils.to_categorical(Y_test, num_classes) 
        Y_test_real = keras.utils.to_categorical(Y_test_real, num_classes)
        Y_test_attack = keras.utils.to_categorical(Y_test_attack, num_classes)

        print("\n" + "after keras.utils.to_categorical()") 

        print ("number of test examples = " + str(X_test.shape[0])) #480
        print ("X_test shape: " + str(X_test.shape))   #(480, 20, 64, 64, 3)
        print ("Y_test shape: " + str(Y_test.shape))   #(480, 20, 2)
        print ("X_test_real shape: " + str(X_test_real.shape))   #(80, 20, 64, 64, 3)
        print ("Y_test_real shape: " + str(Y_test_real.shape))   #(80, 20, 2)
        print ("X_test_attack shape: " + str(X_test_attack.shape))   #(400, 20, 64, 64, 3)
        print ("Y_test_attack shape: " + str(Y_test_attack.shape))   #(400, 20, 2)

        return X_test, Y_test, X_test_real, Y_test_real, X_test_attack, Y_test_attack
