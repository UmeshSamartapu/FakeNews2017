#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fake news detection using an LSTM model
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing import sequence
import os
import getEmbeddings
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt
from collections import Counter

# Check TensorFlow version
print(f"Using TensorFlow version: {tf.__version__}")

top_words = 5000
epoch_num = 5
batch_size = 64

def plot_cmat(y_true, y_pred):
    """Plotting confusion matrix"""
    skplt.plot_confusion_matrix(y_true, y_pred)
    plt.show()

# Read the text data
if not os.path.isfile('./xtr_shuffled.npy') or \
    not os.path.isfile('./xte_shuffled.npy') or \
    not os.path.isfile('./ytr_shuffled.npy') or \
    not os.path.isfile('./yte_shuffled.npy'):
    getEmbeddings.clean_data()

# ✅ Fix: Added allow_pickle=True to avoid ValueError
xtr = np.load('./xtr_shuffled.npy', allow_pickle=True)
xte = np.load('./xte_shuffled.npy', allow_pickle=True)
y_train = np.load('./ytr_shuffled.npy', allow_pickle=True)
y_test = np.load('./yte_shuffled.npy', allow_pickle=True)

cnt = Counter()
x_train = []
for x in xtr:
    x_train.append(x.split())
    for word in x_train[-1]:
        cnt[word] += 1  

# Storing most common words
most_common = cnt.most_common(top_words + 1)
word_bank = {word: idx + 1 for idx, (word, _) in enumerate(most_common)}

# Encode the sentences
for news in x_train:
    news[:] = [word_bank[word] for word in news if word in word_bank]

# Convert labels to lists
y_train = list(y_train)
y_test = list(y_test)

# Remove short news articles
x_train, y_train = zip(*[(x, y) for x, y in zip(x_train, y_train) if len(x) > 10])

# Processing test data
x_test = [x.split() for x in xte]

# Encode test sentences
for news in x_test:
    news[:] = [word_bank[word] for word in news if word in word_bank]

# Truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

# Convert labels to NumPy arrays
y_train = np.array(y_train)
y_test = np.array(y_test)

# Create the LSTM model
embedding_vector_length = 32
model = Sequential([
    Embedding(top_words + 2, embedding_vector_length, input_length=max_review_length),
    LSTM(100),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch_num, batch_size=batch_size)

# Evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy= {scores[1] * 100:.2f}%")

# Predict labels
y_pred = (model.predict(X_test) > 0.5).astype("int32").flatten()

# Draw the confusion matrix
plot_cmat(y_test, y_pred)
