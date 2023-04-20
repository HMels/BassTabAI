# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 12:54:18 2023

@author: Mels

From https://www.tensorflow.org/text/tutorials/text_generation
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import pickle
import os


#%% Read data
# Load the list from the Pickle file
with open('BasslineLibrary.pickle', 'rb') as f:
    BasslineLibrary = pickle.load(f)    
    # chars = BasslineLibrary.Data                           # already vectorised 
    # ids_from_chars = BasslineLibrary.library[chars]
    # chars_from_ids = BasslineLibrary.inverse_library(ids)

# Load the pre-trained embeddings
embedding_weights = np.load('Embeddings.npy')


#%% Create training exampless and targets
## TODO when using window size, use a different input and output length
def preprocess_split(Tokens, window_size, step):
    '''
    Puts the input data in the correct format. 

    Parameters
    ----------
    Tokens : BasslineLibrary.Data list
        A list of the bassline Tokens. This list has already used a dictionary to uniquely 
        describe all possible actions within the dataset.
    window_size : int
        The size of the window / the maximum amount of inputs the model may use to predict
        the output.
    step : int
        The amount of steps in between inputs. This should be smaller than the window_size
        to prevent loss of data.

    Returns
    -------
    Inputs : (N x window_size x dict_size) bool matrix
        Sparse matrix that represents the inputs on which the targets will be trained.
    Targets : (N x dict_size) bool matrix
        Sparse matrix that represents the targets or the next note that will be played.
    dict_size : int
        The size of the dictionary.

    '''    
    # create list of inputs and targets
    input_, target_ = [], []
    dict_size = 0
    for Token in Tokens:
        input_tokens = tf.boolean_mask(Token, Token!=0)
        
        # We generate a temporary dictionary that is smaller than the original. This works
        # because the first so-many dictionary inputs correspond to the ones that can be found 
        # in the first so-many tabs. As we cut off tabs above a certain number, we can therefore
        # also cut off part of the dictionary. 
        dict_size = max(dict_size, max(input_tokens.numpy())+1)
        for i in range(0, input_tokens.shape[0] - window_size, step):
            input_.append(input_tokens[i: i + window_size].numpy().tolist())
            target_.append(input_tokens[i + window_size].numpy().tolist())
            
    return  tf.convert_to_tensor(input_),  tf.convert_to_tensor(target_), dict_size
        
    '''
    # Create matrices from the inputs and targets
    Inputs = np.zeros((len(input_), window_size, dict_size), dtype=bool)
    targets = np.zeros((len(target_), dict_size), dtype=bool)
    for i, input_i in enumerate(input_):
        for j, input_ij in enumerate(input_i):
            Inputs[i, j, input_ij] = 1
        targets[i, target_[i]] = 1
        
    return Inputs, targets, dict_size
    '''

window_size = 16
Input_data, target_data, dict_size = preprocess_split(BasslineLibrary.Data[:800], window_size, step=4)


#%% create validation data
def split_data(Input_data, target_data, split_ratio=0.8, seed=None):
    """
    Randomly splits the input and target tensors into training and validation sets.

    Args:
    - Input_data: A tensor containing the input data.
    - target_data: A tensor containing the target data.
    - split_ratio: The ratio of data to use for training. The rest will be used for validation. Default is 0.8.
    - seed: Optional random seed for reproducibility.

    Returns:
    - A tuple containing the training and validation datasets, each as a tuple of (Input_data, target_data).
    """
    num_elements = tf.shape(Input_data)[0].numpy()

    # Generate a random permutation of the indices
    if seed is not None:
        indices = tf.random.shuffle(tf.range(num_elements), seed=seed)
    else:
        indices = tf.random.shuffle(tf.range(num_elements))

    # Determine the split point
    split_point = int(num_elements * split_ratio)

    # Split the indices into training and validation indices
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    # Create the training and validation datasets
    return ( tf.gather(Input_data, train_indices), tf.gather(target_data, train_indices), 
            tf.gather(Input_data, val_indices), tf.gather(target_data, val_indices) )


Inputs, targets, Inputs_val, targets_val = split_data(Input_data, target_data, split_ratio=.5, seed=None)
validation_data = (Inputs_val, tf.one_hot(targets_val, dict_size))


#%% build the model
class MyModel(tf.keras.Model):
  def __init__(self, dict_size, embedding_weights, rnn_units):
    super().__init__(self)
    # input should be (BATCH_SIZE, SEQUENCE_LENGTH) 
    # so vectorize doesn't mean actually creating a vector of size (BATCH_SIZE, SEQUENCE_LENGTH, dict_size)
    #self.embedding = tf.keras.layers.Embedding(dict_size, embedding_weights.shape[1])
    self.embedding =  layers.Embedding(
        input_dim=embedding_weights.shape[0], 
        output_dim=embedding_weights.shape[1], 
        weights=[embedding_weights], 
        trainable=False,
        name='embedding'
    )
    self.conv1d = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.flatten =tf.keras.layers.Flatten()
    self.dense = tf.keras.layers.Dense(dict_size, activation='softmax')

  @tf.function
  def call(self, inputs, states=None, return_state=False, training=False):
    embedding = self.embedding(inputs, training=False)
    conv1d = self.conv1d(embedding)
    #if states is None:
    #  states = self.gru.get_initial_state(conv1d)
    #gru, states = self.gru(conv1d, initial_state=states, training=training)
    flatten = self.flatten(conv1d)
    outputs = self.dense(flatten, training=training)

    if return_state:
      return outputs, states
    else:
      return outputs


model = MyModel(dict_size, embedding_weights, rnn_units = 64)
model.compile(optimizer='adam', loss='categorical_crossentropy')


##TODO find a way to use rhe model of autofillBassTab together with the states model


#%% Train the model
def plot_learning_curve(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss'] # new line
    epochs = [i for i, _ in enumerate(loss)]
    plt.plot(epochs, loss, color='skyblue', label='Training Loss') # changed to plot
    plt.plot(epochs, val_loss, color='red', label='Validation Loss') # new line
    plt.xlabel('Epochs'); plt.ylabel('Cross Entropy Loss')
    plt.xlim([0,history.params['epochs']])
    plt.ylim(0)
    plt.legend() # new line
    plt.show()

 
'''
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
'''
EPOCHS = 1
batch_size = 128

history = model.fit(Inputs,tf.one_hot(targets, dict_size), epochs=EPOCHS, 
                    batch_size=batch_size,
                    validation_data=validation_data)#, callbacks=[checkpoint_callback])

model.summary()
plot_learning_curve(history)


#%% Initialize generation model
class OneStep(tf.keras.Model):
  def __init__(self, model, dict_size, window_size, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.window_size = window_size
    self.dict_size = dict_size

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = [[0]]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[dict_size])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):    
    # predicted_logits.shape is [batch, char, next_char_logits]
    inputs_padded = self.pad_or_slice(inputs)
    predicted_logits, states = self.model(inputs_padded, states=states,
                                          return_state=True)

    # Only use the last prediction.
    predicted_logits = predicted_logits#[:, -1,:]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "0" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)
    
    return predicted_ids, states

  @tf.function
  def pad_or_slice(self, input_tensor):
    # this function puts the inputs in the correct format
    window_size = self.window_size
    N = tf.shape(input_tensor)[1]
    if N < window_size:                     # Pad the tensor with zeros along the second dimension
        pad_width = window_size - N         # Calculate the amount of padding needed
        paddings = tf.zeros([1,pad_width], dtype=tf.int64)
        return tf.concat([paddings, input_tensor], axis=1)
    else: # slice to the right format
        return input_tensor[:, -window_size:]

one_step_model = OneStep(model, dict_size, window_size, temperature=0)


#%% Generate bassline
import time 

start = time.time()
states = None
result = tf.Variable(BasslineLibrary.Data[50][:8][None,:].numpy(), dtype=tf.int64)#tf.Variable([[0]], dtype=tf.int64)

for n in range(30):
  next_char, _ = one_step_model.generate_one_step(result, states=states)
  result = tf.concat([result, next_char[None]], axis=1)
  
bassline=result

end = time.time()
print("\nGenerated Bassline")
BasslineLibrary.print_detokenize(np.append(bassline, [0]))
print('\nRun time:', end - start)


#%% Export generator
#tf.saved_model.save(one_step_model, 'generateBassTab')
#one_step_model_reloaded = tf.saved_model.load('generateBassTab')
