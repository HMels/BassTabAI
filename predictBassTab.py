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
class PredictBassTab(tf.keras.Model):
    def __init__(self, embedding_weights):
        super(PredictBassTab, self).__init__()
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
        self.output_layer = tf.keras.layers.Dense(128, activation='softmax')

    @tf.function
    def call(self, inputs):
        embedding = self.embedding_layer(inputs)
        flattened = self.flatten_layer(embedding)
        dense1 = self.dense_layer1(flattened)
        dropout1 = self.dropout_layer1(dense1)
        dense2 = self.dense_layer2(dropout1)
        dropout2 = self.dropout_layer2(dense2)
        output = self.output_layer(dropout2)
        return output
'''

class PredictBassTab(tf.keras.Model):
    def __init__(self, embedding_weights):
        super(PredictBassTab, self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=embedding_weights.shape[0], 
            output_dim=embedding_weights.shape[1], 
            weights=[embedding_weights], 
            trainable=False,
            name='embedding'
        )
        self.lstm_layer1 = tf.keras.layers.LSTM(256, return_sequences=True)
        self.dropout_layer1 = tf.keras.layers.Dropout(0.5)
        self.lstm_layer2 = tf.keras.layers.LSTM(128, return_sequences=True)
        self.dropout_layer2 = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(embedding_weights.shape[0], activation='softmax')

    @tf.function
    def call(self, inputs, max_sequence_length):
        embedding = self.embedding_layer(inputs)
        lstm1 = self.lstm_layer1(embedding)
        dropout1 = self.dropout_layer1(lstm1)
        lstm2 = self.lstm_layer2(dropout1)
        dropout2 = self.dropout_layer2(lstm2)
        outputs = []
        for i in range(max_sequence_length):
            output = self.output_layer(dropout2[:, i, :])
            outputs.append(output)
        return tf.stack(outputs, axis=1)


def train_model(Tokens, embedding_weights, epochs=10, batch_size=128, num_batches_perstep=10, batch_stop=None, learning_rate=0.001):
    """
    Trains a neural network using the pre-trained word embeddings on a list of tokenized inputs and targets.

    Parameters:
        inputs (list): A list of tokenized inputs.
        targets (list): A list of targets to predict.
        embedding_weights (numpy array): A numpy array of shape (vocab_size, embedding_size) containing the pre-trained word embeddings.
        epochs (int): The number of epochs to train for (default 10).
        batch_size (int): The batch size to use for training (default 128).
        num_batches_perstep (int): The amount of batches  it will optimize per time.
        batch_stop (int): The maximum amounts of batches to be processed. If None, it will not be triggered. (default None)
        learning_rate (float): The learning rate to use for optimization (default 0.001).

    Returns:
        A trained TensorFlow model.
    """
    # Initialize the model and loss function.
    if batch_stop is None: batch_stop=int(len(Tokens)/batch_size/num_batches_perstep)
    global model
    model = PredictBassTab(embedding_weights)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # iterate over the batchers
    step_batch=0
    for i in range(0, len(Tokens), batch_size*num_batches_perstep):
        if step_batch==batch_stop: break
        print("Training Model - Batch: "+str(step_batch)+"/"
              +str(min(np.round(len(Tokens)/batch_size/num_batches_perstep),batch_stop)))
        step_batch+=1
        
        # Prepare the training dataset.
        inputs_ = []
        targets_ = []
        batch_tokens = Tokens[i:min(i+batch_size*num_batches_perstep, len(Tokens)-1)]
        for input_tokens in batch_tokens:
            #input_tokens = tf.boolean_mask(Token, Token!=1)
            for i in range(input_tokens.shape[0]):
            #    inputs_.append(input_tokens[i])
            #    targets_.append(input_tokens[i])
                inputs_.append(input_tokens[:i])
                targets_.append(input_tokens[i])
        inputs_ = np.array(inputs_)
        targets_ = np.array(targets_)
        dataset = tf.data.Dataset.from_tensor_slices((inputs_, targets_)).shuffle(buffer_size=len(inputs_)).batch(batch_size)
    
        # Train the model.
        for epoch in range(epochs):
            epoch_loss = 0
            for input_batch, target_batch1 in dataset:
                with tf.GradientTape() as tape:
                    output = model(input_batch,1)
                    target_batch = tf.one_hot(tf.squeeze(target_batch1), depth=output.shape[1])
                    loss = tf.reduce_mean(tf.square(output - target_batch))#     tf.keras.losses.CategoricalCrossentropy(output, target_batch)  #           
        
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                epoch_loss += loss
            print('Epoch:',epoch+1, 'Loss:', epoch_loss.numpy())
            
        # save model per step to prevent data to be lost after crash
        model.save('PredictBassTab_temp')
            

#%% loading model
# Load the list from the Pickle file
with open('BasslineLibrary.pickle', 'rb') as f:
    BasslineLibrary = pickle.load(f)

# Load the pre-trained embeddings
embedding_weights = np.load('Embeddings.npy')

# Train the model.
train_model(BasslineLibrary.Data, embedding_weights, batch_stop=1)
model.save('PredictBassTab_temp')


#%% load model
from tensorflow import keras
model = keras.models.load_model('PredictBassTab')


# Generate a sequence of predicted outputs
input_sequence = [1, 5, 10, 0]  # Example input sequence
input_sequence = np.array([input_sequence])
max_sequence_length = 10  # Maximum length of output sequence
predicted_outputs = model.predict(input_sequence, max_sequence_length)

# Convert predicted outputs to integer token sequence
predicted_sequence = np.argmax(predicted_outputs, axis=-1)
predicted_sequence = predicted_sequence[0].tolist()

# Print the predicted sequence
BasslineLibrary.print_detokenize(predicted_sequence)

#BasslineLibrary.print_detokenize(np.concatenate([test_input,prediction,[0]]))

'''
#%% Testing
i = 120
input_data = BasslineLibrary.Data[i]
index_bar = np.where(input_data==0)[0][1]//2

# Use the trained model to make predictions.
test_input = np.array(input_data.numpy()[:index_bar])
output = model.predict(test_input)

prediction = tf.argmax(output, axis=1, output_type=tf.int32).numpy()

print("Input first half of",BasslineLibrary.names[i]+':')
BasslineLibrary.print_detokenize(input_data)

print("Generated by the model")
BasslineLibrary.print_detokenize(np.concatenate([test_input,prediction,[0]]))
'''

'''
The issue you're facing is likely due to the model not being complex enough to learn the patterns in the data.
 This can happen when the model is too simple or the training dataset is too small or noisy.
 Here are some steps you can take to address this issue:

    1. Increase the complexity of your model: Try adding more layers, increasing the number of units per layer,
    or changing the activation functions. You can also try using a pre-trained model and fine-tune it on your data.

    2. Use a larger training dataset: If possible, try to increase the size of your training dataset. You can also
    try to generate more data by adding noise or perturbations to your existing data.

    3. Preprocess your data: Preprocessing your data can help remove noise and make the patterns more apparent.
    You can try using techniques such as normalization, data augmentation, or feature selection.

    4. Tune hyperparameters: Try experimenting with different learning rates, batch sizes, and optimizer settings
    to see if it improves the training process.

    5. Train for longer: If the model is not learning patterns quickly, you may need to train for longer. Try
    increasing the number of epochs and monitor the loss over time to see if the model is still improving.

    6. Monitor the training process: Keep track of the training loss and accuracy to see if the model is making
    progress. You can also monitor the model's performance on a validation dataset to see if it's overfitting
    or underfitting.
'''