import pandas as pd
import numpy as np
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
    def deepLearning(self, learningRate, numberFeature, X_train, Y_train, X_validation, Y_validation, epochs, batch_size):
        model = Sequential(name="DNN")

        model.add(Dense(5, input_dim=int(numberFeature), activation='relu', name='dense_1'))
        model.add(Dense(6, input_dim=5, activation='relu', name='dense_2'))
        model.add(Dense(8, input_dim=6, activation='relu', name='dense_3'))
        model.add(Dense(5, input_dim=8, activation='relu', name='dense_4'))
        model.add(Dense(1, input_dim=5, activation='sigmoid', name='dense_5'))

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
        plt.axis([0, 700, 0, 1])
        plt.legend()
        plt.savefig("loss")
        plt.show()
        print(history.history['accuracy'])
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Training curve')
        plt.ylabel('Accuracy')
        plt.xlabel('No. epoch')
        plt.axis([0, 700, 0, 1])
        plt.legend()
        plt.savefig("accuracy")
        plt.show()

        Y_pred = model.predict(X_train)
        Y_pred = (Y_pred > 0.5)
        confusion_matrix1 = confusion_matrix(Y_train, Y_pred)
        TN = confusion_matrix1[0][0]
        TP = confusion_matrix1[1][1]
        FP = confusion_matrix1[0][1]
        FN = confusion_matrix1[1][0]
        # if(confusion_matrix):
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        print("train data: ", accuracy)

        Y_pred = model.predict(X_validation)
        Y_pred = (Y_pred > 0.5)
        confusion_matrix2 = confusion_matrix(Y_validation, Y_pred)
        TN = confusion_matrix2[0][0]
        TP = confusion_matrix2[1][1]
        FP = confusion_matrix2[0][1]
        FN = confusion_matrix2[1][0]
        # if(confusion_matrix):
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        print("validation data: ",accuracy)

        Y_pred = model.predict(X)
        Y_pred = (Y_pred > 0.5)
        confusion_matrix3 = confusion_matrix(Y, Y_pred)
        TN = confusion_matrix3[0][0]
        TP = confusion_matrix3[1][1]
        FP = confusion_matrix3[0][1]
        FN = confusion_matrix3[1][0]
        # if(confusion_matrix):
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        print("all data: ",accuracy)
        print(TP, TN)

        df = pd.DataFrame(columns=['idx', 'status'])
        i = 0
        for pred in Y_pred:
            if pred[0]:
                df = df.append({"idx": i, "status": "phishing"}, ignore_index=True)
            else:
                df = df.append({"idx": i, "status": "legitimate"}, ignore_index=True)
            i += 1
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
# Loading it into a numpy array
X_test = X_test.to_numpy()

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.01, random_state=42)
dl = DeepLearning()
dl.deepLearning(0.001, numberFeature, X_train, Y_train, X_validation, Y_validation, 700, 64)

