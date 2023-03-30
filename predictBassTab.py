# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:16:14 2023

@author: Mels
"""
import pickle
import tensorflow as tf
import numpy as np

#from Tab2Vec import SkipGramModel
'''
class MyModel(tf.keras.Model):
    def __init__(self, embedding_weights, input_sequence, output_sequence):
        super(MyModel, self).__init__()
        self.input_sequence = input_sequence
        self.output_sequence = output_sequence
        #expanded_weights = tf.tile(tf.expand_dims(embedding_weights, axis=0), [4, embedding_weights.shape[0], embedding_weights.shape[1]])
        #self.embedding_layer = tf.keras.layers.Embedding(
        #    input_dim=embedding_weights.shape[0], 
        #    output_dim=embedding_weights.shape[1], 
        #    weights=embedding_weights, 
        #    #input_length=input_sequence,
        #    trainable=False,
        #    name='embedding'
        #)
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout_layer1 = tf.keras.layers.Dropout(0.5)
        self.dense_layer2 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout_layer2 = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(output_sequence, activation='softmax')

    def call(self, inputs):
        #embedding = self.embedding_layer(inputs)
        flattened = self.flatten_layer(inputs)
        dense1 = self.dense_layer1(flattened)
        dropout1 = self.dropout_layer1(dense1)
        dense2 = self.dense_layer2(dropout1)
        dropout2 = self.dropout_layer2(dense2)
        output = self.output_layer(dropout2)
        #output = tf.argmax(output, axis=1, output_type=tf.int32) # convert to integers
        return output
'''


class MyModel(tf.keras.Model):
    def __init__(self, output_sequence):
        super(MyModel, self).__init__()
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout_layer1 = tf.keras.layers.Dropout(0.5)
        self.dense_layer2 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout_layer2 = tf.keras.layers.Dropout(0.5)
        self.lstm_layer = tf.keras.layers.LSTM(units=64, return_sequences=True)
        self.output_layer = tf.keras.layers.Dense(units=output_sequence, activation='softmax')

    def call(self, inputs):
        flattened = self.flatten_layer(inputs)
        dense1 = self.dense_layer1(flattened)
        dropout1 = self.dropout_layer1(dense1)
        dense2 = self.dense_layer2(dropout1)
        dropout2 = self.dropout_layer2(dense2)
        lstm = self.lstm_layer(dropout2)
        output = self.output_layer(lstm)
        return output


def train_model(Tokens, sequence_length_input=1, sequence_length_output=1, epochs=10, batch_size=128, learning_rate=0.001):
    """
    Trains a neural network using the pre-trained word embeddings on a list of tokenized inputs and targets.

    Parameters:
        inputs (list): A list of tokenized inputs.
        targets (list): A list of targets to predict.
        embedding_weights (numpy array): A numpy array of shape (vocab_size, embedding_size) containing the pre-trained word embeddings.
        vocab_size (int): The size of the vocabulary to use.
        embedding_size (int): The size of the embedding vectors.
        sequence_length_input (int): The number of tokens to use as input (default 4).
        sequence_length_output (int): The number of tokens to predict (default 1).
        epochs (int): The number of epochs to train for (default 10).
        batch_size (int): The batch size to use for training (default 128).
        learning_rate (float): The learning rate to use for optimization (default 0.001).

    Returns:
        A trained TensorFlow model.
    """
    # Initialize the model and loss function.
    model = MyModel(sequence_length_output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    #loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()


    # Prepare the training dataset.
    inputs_ = []
    targets_ = []
    for input_tokens in Tokens:
        for i in range(sequence_length_input, input_tokens.shape[0] - sequence_length_output):
            inputs_.append(input_tokens[i-sequence_length_input:i])
            targets_.append(input_tokens[i:i+sequence_length_output])
    inputs_ = np.array(inputs_)
    targets_ = np.array(targets_)
    dataset = tf.data.Dataset.from_tensor_slices((inputs_, targets_)).shuffle(buffer_size=len(inputs_)).batch(batch_size)

    # Train the model.
    for epoch in range(epochs):
        epoch_loss = 0
        for input_batch, target_batch1 in dataset:
            with tf.GradientTape() as tape:
                output = model(input_batch)
                #output = output[:, -sequence_length_output:, :]
                print("output=",output)
                target_batch = tf.one_hot(tf.squeeze(target_batch1), depth=output.shape[1])
                print("inputbatch=",target_batch)
                loss = tf.reduce_mean(tf.square(output - target_batch))#     tf.keras.losses.CategoricalCrossentropy(output, target_batch)  #           
    
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss += loss
        print('Epoch:',epoch+1, 'Loss:', epoch_loss)

    return model


#%% loading model
# Load the list from the Pickle file
with open('tokenized_inputs.pickle', 'rb') as f:
    BasslineLibrary = pickle.load(f)

# Load the pre-trained embeddings
#embedding_weights = np.load('Embeddings.npy')

# Train the model.
model = train_model(BasslineLibrary.Data, sequence_length_input=1, sequence_length_output=1)


#%% Testing
# Use the trained model to make predictions.
test_input = [0, 1, 1, 1, 5, 1, 1]  # a tokenized input to predict
test_input = np.array(test_input)
output = model.predict(test_input)

prediction = tf.argmax(output, axis=1, output_type=tf.int32).numpy()

BasslineLibrary.print_detokenize(np.concatenate([test_input,prediction]))


