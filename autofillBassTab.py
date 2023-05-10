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
import matplotlib.pyplot as plt
import pickle

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
Input_data, target_data, dict_size = preprocess_split(BasslineLibrary.Data[:5000], window_size, step=4)
##TODO make it predict the 4 next steps

#%% create validation data
def batch_shuffle(num_elements, seed=None, BATCH_SIZE=64):
    '''
    Shuffles the indices of an array per batch

    Parameters
    ----------
    num_elements : int
        The length of the array that will be shuffled.
    seed : int, optional
        The random seed of the shuffle. The default is None.
    BATCH_SIZE : int, optional
        The size of the batches that need to stay intact. The default is 64.

    Returns
    -------
    indices : array int
        The array of shuffled indices.

    '''
    indices = tf.range(num_elements, dtype=tf.int32)
    
    num_batches = num_elements // BATCH_SIZE
    remaining_samples = num_elements % BATCH_SIZE
    
    if seed is not None: # the reshuffling in batch sizes
        batch_indices = tf.random.shuffle(tf.range(num_batches), seed=seed)
    else:
        batch_indices = tf.random.shuffle(tf.range(num_batches))
    
    if remaining_samples > 0:    # Split the shuffled indices into batches with the given batch size
        indices = tf.reshape(indices[:num_batches * BATCH_SIZE], [num_batches, BATCH_SIZE])
        indices = tf.gather(indices, batch_indices)
        remaining_indices = indices[-1, :remaining_samples]
        indices = tf.reshape(indices, [num_elements-remaining_samples])
        split = np.random.randint(num_batches)*BATCH_SIZE
        indices = tf.concat([indices[:split], remaining_indices, indices[split:]], axis=0)
    else:
        indices = tf.reshape(indices, [num_batches, BATCH_SIZE])
        indices = tf.gather(indices, batch_indices)
        indices = tf.reshape(indices, [num_elements])

    return indices
    

def split_data(Input_data, target_data, split_ratio=0.8, seed=None, BATCH_SIZE=32):
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
    indices = batch_shuffle(num_elements, seed, BATCH_SIZE=16)
    
    # Determine the split point
    split_point = int(num_elements * split_ratio)
    
    # Split the indices into training and validation indices
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    # Create the training and validation datasets
    return ( tf.gather(Input_data, train_indices), tf.gather(target_data, train_indices), 
            tf.gather(Input_data, val_indices), tf.gather(target_data, val_indices) )


Inputs, targets, Inputs_val, targets_val = split_data(Input_data, target_data, split_ratio=.8, seed=None, BATCH_SIZE=window_size)
validation_data = (Inputs_val, tf.one_hot(targets_val, dict_size))


#%% building the model
class MyModel(tf.keras.Model):
  def __init__(self, dict_size, embedding_weights, rnn_units):
    super().__init__(self)
    # input should be (BATCH_SIZE, SEQUENCE_LENGTH) 
    # so vectorize doesn't mean actually creating a vector of size (BATCH_SIZE, SEQUENCE_LENGTH, dict_size)
    self.embedding =  layers.Embedding(
        input_dim=embedding_weights.shape[0], 
        output_dim=embedding_weights.shape[1], 
        #weights=[embedding_weights], 
        #trainable=False,
        name='embedding'
    )
    self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
    self.flatten =tf.keras.layers.Flatten()
    self.dense = tf.keras.layers.Dense(dict_size, activation='softmax')

  @tf.function
  def call(self, inputs, states=None, return_state=False, training=False):
    embedding = self.embedding(inputs, training=True)
    if states is None:
      states = self.gru.get_initial_state(embedding)
    gru, states = self.gru(embedding, initial_state=states, training=training)
    flatten = self.flatten(gru)
    outputs = self.dense(flatten, training=training)

    if return_state:
      return outputs, states
    else:
      return outputs

model = MyModel(dict_size, embedding_weights, rnn_units = 128)
model.compile(optimizer='adam', loss='categorical_crossentropy')


#%% Train model
def plot_learning_curve(history):
    loss = history.history['loss']
    epochs = [i for i, _ in enumerate(loss)]
    plt.plot(epochs, loss, color='skyblue', label='Training Loss') # changed to plot
    try: 
        val_loss = history.history['val_loss']
        plt.plot(epochs, val_loss, color='red', label='Validation Loss') 
    except: print("no loss value")
    plt.xlabel('Epochs'); plt.ylabel('Cross Entropy Loss')
    plt.xlim([0,history.params['epochs']])
    plt.ylim(0)
    plt.legend() 
    plt.show()

BATCH_SIZE=128
EPOCHS=4
history = model.fit(Inputs, tf.one_hot(targets, dict_size), epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=validation_data)
plot_learning_curve(history)
##TODO does the mixing of datasets before not negate the whole states thing? Is this actually part of training? How does the source do it?

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
    if temperature!=0:
        log_pred = np.log(prediction) / temperature
        exp_pred = np.exp(log_pred)
        final_pred = exp_pred / np.sum(exp_pred)
        random_pred = np.random.multinomial(1, final_pred)
        return np.argwhere(random_pred)
    else: # No randomness allowed, return the index of the most probable token
        final_pred = np.zeros(prediction.shape[0])
        final_pred[np.argmax(prediction)]=1
        return np.argwhere(final_pred)


def generate_bassline(model, BasslineLibrary, seed, dict_size, input_num=8, temperature=.5, max_len=60, states=None):
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
    input_tokens = BasslineLibrary.Data[seed]
    input_tokens = tf.boolean_mask(input_tokens, input_tokens!=0)[:input_num]

    # iteratively predict the next tokens till the bar if filled
    iter_num = BasslineLibrary.Data[seed].shape[0] - input_num -1
    for i in range(iter_num):
        pred1, states = model(input_tokens[None,i:input_num+i], states=states, return_state=True)
        prediction = random_predict(pred1[0,:], temperature)
        input_tokens = np.concatenate([input_tokens,prediction[0]])
        temperature=temperature*.997
        
    print("\nPredicted Bassline")
    BasslineLibrary.print_detokenize(np.append(input_tokens, [0]))

seed=50
# print the input bassline
print("\nReference Bassline - Song: "+BasslineLibrary.names[seed])
BasslineLibrary.print_detokenize(BasslineLibrary.Data[seed])

print("\nInput Bassline - Song: "+BasslineLibrary.names[seed])
BasslineLibrary.print_detokenize(BasslineLibrary.Data[seed][:window_size])

for _ in range(3):
    generate_bassline(model, BasslineLibrary, seed, dict_size, temperature=.99, input_num=window_size)