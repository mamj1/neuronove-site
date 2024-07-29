import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt



# nacteni datasetu
data_amazon = pd.read_csv('/path/to/amazon_cells_labelled.txt', names=['text', 'sentiment'], sep='\t')
data_imdb = pd.read_csv('/path/to/imdb_labelled.txt', names=['text', 'sentiment'], sep='\t')
data_yelp = pd.read_csv('/path/to/yelp_labelled.txt', names=['text', 'sentiment'], sep='\t')

# zkombinovani datasetu
data = pd.concat([data_amazon, data_imdb, data_yelp])

# zpracovani
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])
data_padded = pad_sequences(sequences, maxlen=100)
labels = tf.keras.utils.to_categorical(data['sentiment'])

# rozdeleni na trenovaci a testovaci data
X_train, X_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.25, random_state=0)

# Model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=100))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

# trenovani modelu
history = model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1, validation_split=0.1)

# Vysledky
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=False)
print(f'Training Accuracy: {train_accuracy:.4f}')
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print(f'Test Accuracy: {accuracy:.4f}')

# Vykresleni
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
