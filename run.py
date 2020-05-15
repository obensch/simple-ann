import tensorflow 
from tensorflow import keras 

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

labels = ["T-Shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

x_train_n = x_train / 255
x_test_n = x_test / 255

x_valid, x_train = x_train_n[:5000], x_train_n[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]
x_test = x_test_n

tensorflow.random.set_seed(77)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["acc"])

model_history = model.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid))

result = model.evaluate(x_test, y_test)
print('Test loss:', result[0])
print('Test accuracy:', result[1])
