import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
from keras import models
from keras import layers
import keras

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
from keras import optimizers

# optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

from tensorflow.keras import optimizers
from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

dataset = "imdb"
X_train = np.load("./" + dataset + "/train.npy", mmap_mode='r')
y_train = np.load("./" + dataset + "/train_label.npy", mmap_mode='r')

# Input for Validation
X_val = X_train[-1000:]
partial_X_train = X_train[:1000]

# Labels for validation
y_val = y_train[-1000:]
partial_y_train = y_train[:1000]
history = model.fit(partial_X_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(X_val, y_val))


history_dict = history.history
history_dict.keys()
import matplotlib.pyplot as plt

# Plotting losses
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label="Training Loss")
plt.plot(epochs, val_loss_values, 'b', label="Validation Loss")

plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.legend()

plt.show()

acc_values = history_dict['binary_accuracy']
val_acc_values = history_dict['val_binary_accuracy']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, acc_values, 'ro', label="Training Accuracy")
plt.plot(epochs, val_acc_values, 'r', label="Validation Accuracy")

plt.title('Training and Validation Accuraccy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Vectorize testing Data
X_test = np.load("./" + dataset + "/test.npy", mmap_mode='r')
y_test = np.load("./" + dataset + "/test_label.npy", mmap_mode='r')

np.set_printoptions(suppress=True)
result = model.predict(X_test)
y_pred = np.zeros(len(result))
for i, score in enumerate(result):
    y_pred[i] = 1 if score > 0.5 else 0

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_pred, y_test)
print(mae)
