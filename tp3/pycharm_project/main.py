import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model

import matplotlib
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Loading the data and checking its dimension
data_train = pd.read_csv('data/train.csv', header=None, skiprows=1)
data_test = pd.read_csv('data/test.csv', header=None, skiprows=1)

# Separating the features and the true value(x,y)
data_train = data_train.drop(0, axis=1)
last_column = data_train.columns[-1]
Y = data_train[last_column]
X = data_train.drop(last_column, axis=1)
# # Loading it into a numpy array
X = X.to_numpy()
Y = Y.to_numpy()
Y = Y == "phishing"
numberFeature = X.shape[1]
# Separating the features and the true value(x,y)
data_test = data_test.drop(0, axis=1)
X_test = data_test
# Loading it into a numpy array
X_test = X_test.to_numpy()

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.1)

# Checking the size

print(X_train.shape)
print(X_validation.shape)
print(Y_train.shape)
print(Y_validation.shape)

model = Sequential(name="DNN")

model.add(Dense(5, input_dim=int(numberFeature), activation='relu', name='dense_1'))
model.add(Dense(6, input_dim=5, activation='relu', name='dense_2'))
model.add(Dense(8, input_dim=6, activation='relu', name='dense_3'))
model.add(Dense(5, input_dim=10, activation='relu', name='dense_4'))
model.add(Dense(1, input_dim=5, activation='sigmoid', name='dense_5'))

opt = SGD(learning_rate=0.001)  # Stochastic Gradient descent

model.compile(loss='binary_crossentropy',  # Definition of the loss
              optimizer=opt,
              metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=700, batch_size=64, verbose=2,
                    validation_data=(X_validation, Y_validation))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training curve')
plt.ylabel('Loss')
plt.xlabel('No. epoch')
plt.axis([0, 2000, 0, 1])
plt.legend()
plt.savefig("loss")
plt.show()

plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Training curve')
plt.ylabel('Accuracy')
plt.xlabel('No. epoch')
plt.axis([0, 2000, 0, 1])
plt.legend()
plt.savefig("accuracy")
plt.show()

Y_pred = model.predict(X_train)
Y_pred = (Y_pred > 0.5)
print(confusion_matrix(Y_train, Y_pred))

Y_pred = model.predict(X_validation)
Y_pred = (Y_pred > 0.5)
print(confusion_matrix(Y_validation, Y_pred))

Y_pred = model.predict(X)
Y_pred = (Y_pred > 0.5)
print(confusion_matrix(Y, Y_pred))

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
