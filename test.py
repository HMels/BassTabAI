# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:10:01 2023

@author: Mels
"""
import pickle
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

tf.config.run_functions_eagerly(True)

class PredictBassTab(tf.keras.Model):
    def __init__(self, embedding_weights):
        super(PredictBassTab, self).__init__()
        self.output_len = 128
        #self.masking_layer = tf.keras.layers.Masking(mask_value=0)
        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=embedding_weights.shape[0], 
            output_dim=embedding_weights.shape[1], 
            weights=[embedding_weights], 
            trainable=False,
            name='embedding'
        )
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout_layer1 = tf.keras.layers.Dropout(0.5)
        self.dense_layer2 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout_layer2 = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(self.output_len, activation='softmax')

    @tf.function
    def call(self, inputs):
        #masked_inputs = self.masking_layer(inputs)
        embedding = self.embedding_layer(inputs)
        flattened = self.flatten_layer(embedding)
        dense1 = self.dense_layer1(flattened)
        dropout1 = self.dropout_layer1(dense1)
        dense2 = self.dense_layer2(dropout1)
        dropout2 = self.dropout_layer2(dense2)
        output = self.output_layer(dropout2)
        return output


def variable_length_generator(data, batch_size, max_seq_len):
    '''
    variable_length_generator is a generator function that generates batches of 
    variable-length input sequences for a neural network. It takes in two arguments,
    data and batch_size, where data is a numpy array of sequences and batch_size is
    the size of the batch to generate.
    
    In each iteration, the function shuffles the data and generates batches of input
    sequences, where each sequence has its length padded to match the length of the 
    longest sequence in the batch. The resulting batch is then returned as a tuple of
    two numpy arrays, where the first array contains the padded input sequences and 
    the second array contains the corresponding target sequences.
    
    The function is designed to be used with the fit method of a Keras model to train
    the model on variable-length input sequences. By using a generator function to 
    generate the batches of input sequences, it allows the model to train on sequences
    of different lengths without having to pre-process the data to make all sequences 
    the same length.
    '''
    while True:
        # Shuffle the data at the start of each epoch
        np.random.shuffle(data)
        
        # Generate batches of variable-length input sequences
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            batch_inputs = []
            batch_targets = []
            #max_seq_len = tf.reduce_max([tf.shape(sequence)[0] for sequence in batch_data])
            for sequence in batch_data:
                input_sequence = sequence[:-1]
                target_sequence = sequence[1:]
                input_sequence = tf.pad(input_sequence, [[0, max_seq_len - tf.shape(input_sequence)[0]]], constant_values=0)
                target_sequence = tf.pad(target_sequence, [[0, max_seq_len - tf.shape(target_sequence)[0]]], constant_values=0)
                batch_inputs.append(input_sequence)
                batch_targets.append(target_sequence)
            yield tf.stack(batch_inputs), tf.stack(batch_targets)


def train_model(data, embedding_weights, batch_size=32, epochs=10):
    # Split the data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # Convert labels to one-hot encoding
    #train_data = to_categorical(train_data, num_classes=128)
    #val_data = to_categorical(val_data, num_classes=128)

    # Create the model
    model = PredictBassTab(embedding_weights)
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train the model using the generator function
    train_generator = variable_length_generator(train_data, batch_size, model.output_len)
    val_generator = variable_length_generator(val_data, batch_size, model.output_len)
    steps_per_epoch = len(train_data) // batch_size
    validation_steps = len(val_data) // batch_size
    model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
              validation_data=val_generator, validation_steps=validation_steps)
    
    return model


#%%
# Load the list from the Pickle file
with open('BasslineLibrary.pickle', 'rb') as f:
    BasslineLibrary = pickle.load(f)

# Load the pre-trained embeddings
embedding_weights = np.load('Embeddings.npy')

#%%
# Initialize the model
model = train_model(BasslineLibrary.Data[:200] , embedding_weights)


#%%
# Generate a sequence of predicted outputs
input_sequence = [1, 5, 10, 0]  # Example input sequence
input_sequence = np.array([input_sequence])
max_sequence_length = 10  # Maximum length of output sequence
predicted_outputs = model.predict(input_sequence, max_sequence_length)

# Convert predicted outputs to integer token sequence
predicted_sequence = np.argmax(predicted_outputs, axis=-1)
predicted_sequence = predicted_sequence[0].tolist()

# Print the predicted sequence
print(predicted_sequence)


#%%
from tensorflow.keras.utils import pad_sequences

i = 120
input_data = BasslineLibrary.Data[i]
index_bar = np.where(input_data==0)[0][1]//2

# Pad your input sequences to the maximum length
padded_input = pad_sequences([input_data.numpy()[:index_bar]], maxlen=32, padding='post', truncating='post')[0]

# Use the trained model to make predictions
output = model.predict(tf.one_hot(padded_input, embedding_weights.shape[1]))


#%% Testing

# Use the trained model to make predictions.
test_input = np.array(input_data.numpy()[:index_bar])
output = model.predict(test_input)

#prediction = tf.argmax(output, axis=1, output_type=tf.int32).numpy()

print("Input first half of",BasslineLibrary.names[i]+':')
BasslineLibrary.print_detokenize(input_data)

print("Generated by the model")
BasslineLibrary.print_detokenize(np.concatenate([test_input,prediction,[0]]))

