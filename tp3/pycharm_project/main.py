import pandas as pd
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model

import matplotlib
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class DeepLearning:
    def __init__(self):
        None
    def model(self, numberFeatures, numberLayers, numberNodes):
        model = Sequential(name="DNN")
        model.add(Dense(numberNodes[0], input_dim=int(numberFeatures), activation='relu', name='dense_0'))
        for i in range(1, numberLayers):

            model.add(Dense(numberNodes[i], input_dim=numberNodes[i-1], activation='relu', name='dense_'+str(i)))
        model.add(Dense(1, input_dim=numberNodes[numberLayers-1], activation='sigmoid', name='dense_'+str(numberLayers)))
        return model
    def deepLearning(self, learningRate, numberFeatures, X_train, Y_train, X_validation, Y_validation, epochs, batch_size, numberLayers, numberNodes):
        model = self.model(numberFeatures,numberLayers, numberNodes)

        opt = SGD(learning_rate=learningRate)  # Stochastic Gradient descent

        model.compile(loss='binary_crossentropy',  # Definition of the loss
                      optimizer=opt,
                      metrics=['accuracy'])
        history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=2,
                            validation_data=(X_validation, Y_validation))

        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training curve')
        plt.ylabel('Loss')
        plt.xlabel('No. epoch')
        plt.axis([0, epochs, 0, 1])
        plt.legend()
        plt.savefig("loss")
        plt.show()
        print(history.history['accuracy'])
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Training curve')
        plt.ylabel('Accuracy')
        plt.xlabel('No. epoch')
        plt.axis([0, epochs, 0, 1])
        plt.legend()
        plt.savefig("accuracy")
        plt.show()

        Y_pred = model.predict(X_train)
        Y_pred = (Y_pred > 0.5)

        # if(confusion_matrix):
        accuracy1 = self.accuracy(Y_train, Y_pred)
        print("train data: ", accuracy1)

        Y_pred = model.predict(X_validation)
        Y_pred = (Y_pred > 0.5)

        # if(confusion_matrix):
        accuracy2 = self.accuracy(Y_validation, Y_pred)
        print("validation data: ",accuracy2)

        Y_pred = model.predict(X)
        Y_pred = (Y_pred > 0.5)

        # if(confusion_matrix):
        accuracy3 = self.accuracy(Y, Y_pred)
        print("all data: ",accuracy3)

        df = pd.DataFrame(columns=['idx', 'status'])
        i = 0
        for pred in Y_pred:
            if pred[0]:
                df = df.append({"idx": i, "status": "phishing"}, ignore_index=True)
            else:
                df = df.append({"idx": i, "status": "legitimate"}, ignore_index=True)
            i += 1
        df = df.append({"idx": "train", "status": accuracy1}, ignore_index=True)
        df = df.append({"idx": "validate", "status": accuracy2}, ignore_index=True)
        df = df.append({"idx": "all", "status": accuracy3}, ignore_index=True)
        df = df.append({"idx": "layers", "status": numberNodes}, ignore_index=True)
        df = df.append({"idx": "epoch", "status": epochs}, ignore_index=True)
        df = df.append({"idx": "batch size", "status": batch_size}, ignore_index=True)
        df.to_csv("data/predictions.csv", index=False)
        Y_pred = model.predict(X_test)
        Y_pred = (Y_pred > 0.5)
        df2 = pd.DataFrame(columns=['idx', 'status'])
        i = 0
        for pred in Y_pred:
            if pred[0]:
                df2 = df2.append({"idx": i, "status": "phishing"}, ignore_index=True)
            else:
                df2 = df2.append({"idx": i, "status": "legitimate"}, ignore_index=True)
            i += 1
        df2.to_csv("data/submission.csv", index=False)
    def accuracy(self, Y_true, Y_pred):
        confusion_matrix = sklearn.metrics.confusion_matrix(Y_true, Y_pred)
        TN = confusion_matrix[0][0]
        TP = confusion_matrix[1][1]
        FP = confusion_matrix[0][1]
        FN = confusion_matrix[1][0]
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        return accuracy


# Loading the data and checking its dimension
data_train = pd.read_csv('data/train.csv', header=None, skiprows=1)
data_test = pd.read_csv('data/test.csv', header=None, skiprows=1)

# Separating the features and the true value(x,y)
data_train = data_train.drop(0, axis=1)
last_column = data_train.columns[-1]
Y = data_train[last_column]
X = data_train.drop(last_column, axis=1)
numberFeature = X.shape[1]
#normalisation of features
print(X.shape)
mean = X.mean()
var = X.var()
for i in range(1,numberFeature+1):

    if(var[i] !=0):
        X[i] = (X[i]-mean[i])/var[i]

# isnull=X.isnull().sum()
# for i in range(1,len(isnull)+1):
#     if(isnull[i]!=0):
#         print(i, isnull[i],var[i])

# Loading it into a numpy array
X = X.to_numpy()
Y = Y.to_numpy()
Y = Y == "phishing"
assert not np.any(np.isnan(X))
# Separating the features and the true value(x,y)
data_test = data_test.drop(0, axis=1)
X_test = data_test

#normalisation of features
mean_test = X_test.mean()
var_test = X_test.var()
for i in range(1,numberFeature+1):

    if(var_test[i] !=0):
        X_test[i] = (X_test[i]-mean_test[i])/var_test[i]

# assert not np.any(np.isnan(X_test))

# Loading it into a numpy array
X_test = X_test.to_numpy()

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.25)
print(len(Y_validation))
dl = DeepLearning()
# bach_sizes = [32, 48, 64, 80, 96, 112, 128]
# epoches = [500, 500, 700, 700, 1000, 1000, 1500]
numberNodes = [5,12,10,5]
numberLayers = 4
dl.deepLearning(0.001, numberFeature, X_train, Y_train, X_validation, Y_validation, 1100, 64, numberLayers, numberNodes)