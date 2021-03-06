from __future__ import print_function

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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


    #We want to have a binary classification: digit 0 is classified 1 and 
    #all the other digits are classified 0

    y_new = np.zeros(y_train.shape)
    y_new[np.where(y_train==0.0)[0]] = 1
    y_train = y_new

    y_new = np.zeros(y_test.shape)
    y_new[np.where(y_test==0.0)[0]] = 1
    y_test = y_new

    return x_train,y_train,x_test,y_test



#Let start our work: creating a neural network
#First, we just use a single neuron. 

#####TO COMPLETE




def plot_curves(history,class_1,act,neuron):
    
    # summarize history for loss
    plt.plot(history.history['loss'],label="Training Loss")
    plt.plot(history.history['val_loss'],label="Validation Loss")
    plt.title('Model Loss')
    plt.title(" Model loss for " +str(neuron)+ " Neuron Binary Classifier ")
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Training loss', 'Validation loss'], loc='upper right')
    
    plt.savefig(
        "losses--binary-classification-{0}-VS- all -{1}-_"+str(act)+" with neuron "+str(neuron)+".png".format(str(class_1),str(epochs)))

    plt.clf()                                                                                                                                                                                                 
 


    # summarize history for  accuracy
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(" Model accuracy for" +str(neuron)+ "Neuron Binary Classifier ")
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['Training accuracy', 'Validation accuracy'], loc='upper right')
    
    plt.savefig(
        "Accuracy--binary-classification-{0}-VS- all -{1}-_"+str(act)+" with neuron "+str(neuron)+".png".format(str(class_1),str(epochs)))

    plt.clf()

def buildNetwork(number_neuron,input_dim,activation_fun):
    model = Sequential()
    # Adding the input layer and the first hidden layer with dropout
    model.add(Dense(units = number_neuron, activation = activation_fun, input_dim =input_dim))
    model.add(Dense(1,activation='sigmoid',input_dim=number_neuron))
    model.compile(optimizer = 'SGD', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

def train_model(model,X,Y,valiation_split_size,lr,batch_size,epochs):
    """
    Training the network

    :return: model history
    """
    history = model.fit(X, Y,validation_split=valiation_split_size, epochs=epochs, batch_size=batch_size)
    
    return history
    



if __name__ == "__main__":

    history=[]
    Accuracy_list=[]
    activation=['relu','sigmoid','tanh']
    number_neuron=[64,128,256]
    # change this to other digits to change 1 vs n
    class_1 = 0
    X_train, Y_train, X_test, Y_test = data()

    # hyperparameters
    in_dim = X_train.shape[1]  # 784
    learning_rate = 0.1
    batch_size = 128
    epochs = 100
    validation_split_size=0.3
    filepath = "./accuracy_excercise_02.txt"
    with open(filepath, 'w') as f:
        for neuron in number_neuron:
            for act in activation:
                #Compile the model 
                model=buildNetwork(neuron,in_dim,act)

                # train model
                history = train_model(model,X_train,Y_train,validation_split_size,learning_rate,batch_size,epochs)

                # plot training loss& accuracy
                plot_curves(history,class_1,act,neuron)

                #Find the accuracy on test data
                _, accuracy = model.evaluate(X_test, Y_test)
                
                print("=== Accuracy with activation function "+str(act)+" and neuron number "+str(neuron)+ "= {:.2f}=== %".format(accuracy * 100))
                Accuracy_list.append(accuracy)
                
            for i,acc in enumerate (activation):
                f.write("Accuracy with activation function ("+acc+") and "+str(neuron)+" Neurons : ")
                f.write( str(Accuracy_list[i]*100))
                f.write("\n")

    print("Done.")


         
            






