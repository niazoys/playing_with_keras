from __future__ import print_function

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt

def data():
        
    #load (first download if necessary) the MNIST dataset
    # (the dataset is stored in your home direcoty in ~/.keras/datasets/mnist.npz
    #  and will take  ~11MB)
    # data is already split in train and test datasets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # x_train : 60000 images of size 28x28, i.e., x_train.shape = (60000, 28, 28)
    # y_train : 60000 labels (from 0 to 9)
    # x_test  : 10000 images of size 28x28, i.e., x_test.shape = (10000, 28, 28)
    # x_test  : 10000 labels
    # all datasets are of type uint8

    #To input our values in our network Dense layer, we need to flatten the datasets, i.e.,
    # pass from (60000, 28, 28) to (60000, 784)
    #flatten images
    num_pixels = x_train.shape[1] * x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], num_pixels)
    x_test = x_test.reshape(x_test.shape[0], num_pixels)

    #Convert to float
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    #Normalize inputs from [0; 255] to [0; 1]
    x_train = x_train / 255
    x_test = x_test / 255


    #Convert class vectors to binary class matrices ("one hot encoding")
    ## Doc : https://keras.io/utils/#to_categorical
    y_train = tensorflow.keras.utils.to_categorical(y_train)
    y_test = tensorflow.keras.utils.to_categorical(y_test)


    num_classes = y_train.shape[1]

    return x_train,y_train,x_test,y_test



#Let start our work: creating a neural network
#First, we just use a single neuron. 

#####TO COMPLETE




def plot_curves(history,act):
    
    # summarize history for accuracy
    plt.plot(history.history['loss'],label="Training Loss")
    plt.plot(history.history['val_loss'],label="Validation Loss")
    plt.title(" Model loss for 64 Neuron Multiclass Classifier ")
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
    
    plt.savefig(
        "training-validation-Loss--multiclass-classification-{1}-epoches_"+str(act)+".png".format(str(epochs)))
   
    plt.clf()
    # summarize history for loss
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(" Model accuracy for 64 Neuron Multiclass Classifier ")
    plt.ylabel('Accuraacy')
    plt.xlabel('epoch')
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper right')
    
    
    plt.savefig(
        "training-validation-Accuracy--multiclass-classification--{1}-epoches"+str(act)+".png".format(str(epochs)))
    plt.clf()


def buildNetwork(input_dim,opt):
    model = Sequential()
    # Adding the input layer and the first hidden layer with dropout
    model.add(Dense(units = 64, activation = 'relu', input_dim =784))
    model.add(Dense(units = 10, activation = 'softmax', input_dim =64))
    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

def train_model(model,X,Y,valiation_split_size,lr,batch_size,epochs):
    """
    Training the network

    :return: model history
    """
    history = model.fit(X, Y,validation_split=valiation_split_size, epochs=epochs, batch_size=batch_size)
    
    return history
    



if __name__ == "__main__":

    #list of optimizers
    optimizer_list=['SGD','ADAM','RMSprop']

    X_train, Y_train, X_test, Y_test = data()
    
    filepath= "./accuracy_excercise_03.txt"
    
    Accuracy_list=[]

    # hyperparameters
    in_dim = X_train.shape[0]  # 784
    learning_rate = 0.1
    epochs = 100
    validation_split_size=0.3
    batch=128

    for opt in optimizer_list:
        #Compile the model 
        model=buildNetwork(in_dim,opt)

        # train model
        history = train_model(model,X_train,Y_train,validation_split_size,learning_rate,batch,epochs)

        # plot training loss& accuracy
        plot_curves(history,opt)

        #Find the accuracy on test data
        _, accuracy = model.evaluate(X_test, Y_test)
        
        print("=== Accuracy with optimizer"+str(opt)+"= {:.2f}=== %".format(accuracy * 100))
        Accuracy_list.append(accuracy)
        
    with open(filepath, 'w') as f:
        for i,b in enumerate (optimizer_list):
            f.write("Accuracy with optimizer ("+str(b)+"): ")
            f.write( str(Accuracy_list[i]*100))
            f.write("\n")

    print("Done.")





