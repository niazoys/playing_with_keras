from __future__ import print_function

#import tensorflow as tf
#import tensorflow.keras
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




def plot_curves(history,class_1,act):
    
    # summarize history for accuracy
    plt.plot(history.history['loss'],label="Training Loss")
    plt.plot(history.history['val_loss'],label="Validation Loss")
    plt.title('model accuracy')
    plt.title(" Model loss for 1 Neuron Binary Classifier ")
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    
    plt.savefig(
        "training-validation-losses--64_neuron-binary-classification-{0}-VS- all -{1}-epoches_"+str(act)+".png".format(str(class_1),str(epochs)))
                                                                                                                                                                                                                 
 


    # summarize history for loss
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(" Model accuracy for 1 Neuron Binary Classifier ")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')
    
    plt.savefig(
        "training-validation-Accuracy--64_neuron-binary-classification-{0}-VS- all -{1}-epoches_"+str(act)+".png".format(str(class_1),str(epochs)))
   


def buildNetwork(input_dim,activation_fun):
    model = Sequential()
    # Adding the input layer and the first hidden layer with dropout
    model.add(Dense(units = 64, activation = activation_fun, input_dim =784))
    model.add(Dense(1,activation='sigmoid',input_dim=64))
    model.compile(optimizer = 'SGD', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

def train_model(model,X,Y,valiation_split_size,lr,batch_size,epochs):
    """
    Training the network

    :return: model history
    """
    history = model.fit(X, Y,validation_split=valiation_split_size, epochs=epochs, batch_size=1)
    
    return history
    



if __name__ == "__main__":

    history=[]
    Accuracy_list=[]
    activation=['relu','sigmoid','tanh']
    # change this to other digits to change 1 vs n
    class_1 = 0
    X_train, Y_train, X_test, Y_test = data()

    # hyperparameters
    in_dim = X_train.shape[0]  # 784
    learning_rate = 0.5
    batch_size = 64
    epochs = 1
    validation_split_size=0.3
    filepath = "./accuracy_excercise_02.txt"

    for act in activation:
        #Compile the model 
        model=buildNetwork(in_dim,act)

        # train model
        history = train_model(model,X_train,Y_train,validation_split_size,learning_rate,batch_size,epochs)

        # plot training loss& accuracy
        plot_curves(history,class_1,act)

        #Find the accuracy on test data
        _, accuracy = model.evaluate(X_test, Y_test)
        
        print("=== Accuracy with "+str(act)+"= {:.2f}=== %".format(accuracy * 100))
        Accuracy_list.append(accuracy)
        
    with open(filepath, 'w') as f:
        for i,acc in enumerate (activation):
            f.write("Accuracy with ("+acc+"): ")
            f.write( str(Accuracy_list[i]*100))
            f.write("\n")

    print("Done.")


         
            






