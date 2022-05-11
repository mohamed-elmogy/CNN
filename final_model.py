import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, MaxPool2D, Conv2D
from keras.datasets import mnist
from sklearn.metrics import classification_report

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)
model = Sequential()
#--------------------------------------------------------------------------------------------------------
model.add(Conv2D(64, (3, 3),input_shape=(28,28,1)))
model.add(Activation("tanh"))
model.add(MaxPool2D(pool_size=(2, 2)))
# -------------------------------------------------------------------------------------------------------
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("tanh"))
# -------------------------------------------------------------------------------------------------------
model.add(Dense(10))
model.add(Activation("softmax"))
# -------------------------------------------------------------------------------------------------------
model.compile(loss="sparse_categorical_crossentropy",optimizer = "SGD",metrics=["accuracy"],)
model.fit(x_train,y_train,64,15, shuffle=True)
#--------------------------------------------------------------------------------------------------------
y_pred = model.predict(x_test)
labels = [np.argmax(i) for i in y_pred]

print(classification_report(y_test, labels))
#--------------------------------------------------------------------------------------------------------
model.summary()