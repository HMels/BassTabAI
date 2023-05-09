# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:00:49 2023

@author: Mels

This model is able to take in a bassline and predict the rest of the bassline from it.
It is based on the model created by https://jaketae.github.io/study/auto-complete/ and it takes uses 
a multilayer model together with some randomness to predict which tab should follow the input tab. 
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import pickle


#%% loading data
# Load the list from the Pickle file
with open('BasslineLibrary.pickle', 'rb') as f:
    BasslineLibrary = pickle.load(f)

# Load the pre-trained embeddings
embedding_weights = np.load('Embeddings.npy')


#%% prepare data for training
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
            input_.append(input_tokens[i: i + window_size].numpy())
            target_.append(input_tokens[i + window_size].numpy())
    return tf.stack(input_), tf.stack(target_), dict_size


window_size = 12
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


#%% building the model #2
class MyModel(tf.keras.Model):
  def __init__(self, dict_size, embedding_weights, rnn_units):
    super().__init__(self)
    # input should be (BATCH_SIZE, SEQUENCE_LENGTH) 
    # so vectorize doesn't mean actually creating a vector of size (BATCH_SIZE, SEQUENCE_LENGTH, dict_size)
    #self.conv1d = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')
    self.embedding =  layers.Embedding(
        input_dim=embedding_weights.shape[0], 
        output_dim=embedding_weights.shape[1], 
        weights=[embedding_weights], 
        trainable=False,
        name='embedding'
    )
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.flatten =tf.keras.layers.Flatten()
    self.dense = tf.keras.layers.Dense(dict_size, activation='softmax')

  @tf.function
  def call(self, inputs, states=None, return_state=False, training=False):
    embedding = self.embedding(inputs, training=False)
    #conv1d = self.conv1d(inputs)
    if states is None:
      states = self.gru.get_initial_state(embedding)
    gru, states = self.gru(embedding, initial_state=states, training=training)
    flatten = self.flatten(gru)
    outputs = self.dense(flatten, training=training)

    if return_state:
      return outputs, states
    else:
      return outputs


model = MyModel(dict_size, embedding_weights, rnn_units = 64)
#model.build(input_shape=(None, window_size))
model.compile(optimizer='adam', loss='categorical_crossentropy')
#model.summary()


#%% Train model
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

history = model.fit(Inputs, tf.one_hot(targets, dict_size), epochs=10, batch_size=128, validation_data=validation_data)
plot_learning_curve(history)


#%% saving the model
#tf.saved_model.save(model, 'autofillBasTab')


#%% loading the model
#import numpy as np
#import tensorflow as tf


#model = tf.saved_model.load('autofillBasTab')


#%% 
def random_predict(prediction, temperature):
    '''
    Algorithm that uses temperature to add randomness to the prediction.

    Parameters
    ----------
    prediction : int array
        Array representing the probabilities for certain tokens to be most likely to occure.
        Has been predicted by the model via model.predict().
    temperature : float
        Temperature is a metric that increases the randomness of the output.

    Returns
    -------
    random_pred : int
        The predicted token, randomised.

    '''
    prediction = np.asarray(prediction).astype('float64')
    if temperature !=0:
        log_pred = np.log(prediction) / temperature
        exp_pred = np.exp(log_pred)
        final_pred = exp_pred / np.sum(exp_pred)
        random_pred = np.random.multinomial(1, final_pred)
        return random_pred
    else:
        # No randomness allowed, return the index of the most probable token
        final_pred = np.zeros(prediction.shape[0])
        final_pred[np.argmax(prediction)]=1
        return final_pred


def generate_bassline(model, BasslineLibrary, seed, dict_size, input_num=8, temperature=.5, max_len=60):
    '''
    Inputs a bassline from a seed with a certain input size, and will try to predict the 
    following notes according to the trained model together with a temperature for randomness

    Parameters
    ----------
    model : keras.model
        The multilayered model that has been trained on the data.
    BasslineLibrary : BasslineLibrary
        The BasslineLibrary class containing the decoding functions and the dictionary.
    seed : int
        Describes which bassline from the data to input.
    input_num : int
        The number of input tokens after which it will predict. The default is 8.
    temperature : float, optional
        Adds randomness to predictions. The default is .5.
    max_len : int, optional
        The maximum length of the input. The default is 60.
    dict_size : int
        The size of the dictionary.

    '''
    # define the input matrix
    input_tokens = BasslineLibrary.Data[seed][:input_num]
    ##TODO add padding here
    bassline = np.zeros([input_num, dict_size], dtype=bool)
    for i, index in enumerate(input_tokens):
        bassline[i, index] = 1

    # iteratively predict the next tokens till the bar if filled
    iter_num = BasslineLibrary.Data[seed].shape[0] - input_num -1
    for i in range(iter_num):
        pred1 = model.predict(bassline[i: i + max_len][None,:,:])[0]
        prediction = random_predict(pred1, temperature).astype(bool)
        bassline = np.concatenate([bassline,prediction[None,:]])
     
    # print the input bassline
    print("\nReference Bassline - Song: "+BasslineLibrary.names[seed])
    BasslineLibrary.print_detokenize(BasslineLibrary.Data[seed])
    
    print("\nInput Bassline - Song: "+BasslineLibrary.names[seed])
    BasslineLibrary.print_detokenize(np.argwhere(bassline)[:input_num,1])
    
    print("\nPredicted Bassline")
    BasslineLibrary.print_detokenize(np.append(np.argwhere(bassline)[:,1], [0]))


generate_bassline(model, BasslineLibrary, 50, dict_size, temperature=1, input_num=window_size)