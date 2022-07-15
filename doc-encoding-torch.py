from keras.datasets import imdb

# Load the data, keeping only 10,000 of the most frequently occuring words
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print(type([max(sequence) for sequence in train_data]))

# Find the maximum of all max indexes
print(max([max(sequence) for sequence in train_data]))
# Let's quickly decode a review

# step 1: load the dictionary mappings from word to integer index
word_index = imdb.get_word_index()

# step 2: reverse word index to map integer indexes to their respective words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Step 3: decode the review, mapping integer indices to words
#
# indices are off by 3 because 0, 1, and 2 are reserverd indices for "padding", "Start of sequence" and "unknown"
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

import numpy as np


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # Creates an all zero matrix of shape (len(sequences),10K)
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1  # Sets specific indices of results[i] to 1s
    return results


len(train_data)
# Vectorize training Data
X_train = vectorize_sequences(train_data[:5000])
y_train = np.asarray(train_labels[:5000]).astype('float32')

from models.torch_models import BinaryClassification, convert_to_train_loader, convert_to_test_loader
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BinaryClassification(10000)
model.to(device)
print(model)

print(device)

# Input for Validation
X_val = X_train[-1000:]
partial_X_train = X_train[:-1000]

# Labels for validation
y_val = y_train[-1000:]
partial_y_train = y_train[:-1000]

print(type(partial_X_train))
print(partial_X_train.shape)
data = convert_to_train_loader(partial_X_train, partial_y_train)
history = model.fit(data, device, EPOCHS=14)
X_test = vectorize_sequences(test_data)
test_data_loader = convert_to_test_loader(X_test)
model.evaluate(test_labels, test_data_loader, device)
