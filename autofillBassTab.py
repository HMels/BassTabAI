# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:26:58 2023

@author: Mels
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pickle


#%% AutofillBassTab
class AutofillBassTab(tf.keras.Model):
    def __init__(self, N=6000, window_size=12, prediction_length=6, step=3, split_ratio=0.75, model_path=None):
        super(AutofillBassTab, self).__init__()
        self.N = N  # Amount of data
        self.window_size = window_size  # Input token length
        self.prediction_length = prediction_length  # Number of tokens to predict per step
        self.step = step  # Number of tokens to jump in training
        self.split_ratio = split_ratio  # Training/validation split
        
        # Initialize the dataset variables
        self.embedding_weights = None
        self.BasslineLibrary = None
        self.Input_data = None # the input data loaded via preprocess_split()
        self.Target_data = None # the Target data as loaded via preprocess_split()
        self.dict_size = None # the temporary dictionary size, as we only load N data
        
        if model_path is not None: 
            self.model = tf.keras.models.load_model(model_path, compile=False)
        else: 
            self.model = None
            
    
    def load_dataset(self, embedding_weights, BasslineLibrary):
        """
        Load the dataset for training the model.
        
        Args:
            embedding_weights: The embedding weights for the dataset.
            BasslineLibrary: The BasslineLibrary dataset.
        """
        self.embedding_weights = embedding_weights
        self.BasslineLibrary = BasslineLibrary
        
        # split data into targets and inputs
        self.Input_data, self.Target_data, self.dict_size = self.preprocess_split()
        
        # split into training and validation and get them in the right format
        self.Input_data, self.Target_data, self.Inputs_val, self.Targets_val = self.split_data(seed=None, BATCH_SIZE=self.window_size)
        self.validation_data = (self.Inputs_val, tf.one_hot(self.Targets_val, self.dict_size))
        self.Target_data = tf.one_hot(self.Target_data, self.dict_size)
        
          
    def preprocess_split(self):
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
        Tokens = self.BasslineLibrary.Data[:self.N]
        for Token in Tokens:
            
            # we mask all the 0's but add one at the end as an EOL
            input_tokens = tf.boolean_mask(Token, Token!=0)
            input_tokens=tf.Variable(input_tokens.numpy(), dtype=tf.int32)
            input_tokens=tf.concat([input_tokens, [0]], axis=0)
            
            # We generate a temporary dictionary that is smaller than the original. This works
            # because the first so-many dictionary inputs correspond to the ones that can be found 
            # in the first so-many tabs. As we cut off tabs above a certain number, we can therefore
            # also cut off part of the dictionary. 
            dict_size = max(dict_size, max(input_tokens.numpy())+1)   
            for i in range(0, input_tokens.shape[0] - self.window_size - self.prediction_length, self.step):
                input_.append(input_tokens[i: i + self.window_size].numpy())
                target_.append(input_tokens[i + self.window_size: i + self.window_size + self.prediction_length].numpy())
        return tf.stack(input_), tf.stack(target_), dict_size


    def batch_shuffle(self, num_elements, seed=None, BATCH_SIZE=64):
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
        

    def split_data(self, seed=None, BATCH_SIZE=32):
        """
        Randomly splits the input and target tensors into training and validation sets.

        Args:
        - Input_data: A tensor containing the input data.
        - Target_data: A tensor containing the target data.
        - split_ratio: The ratio of data to use for training. The rest will be used for validation. Default is 0.8.
        - seed: Optional random seed for reproducibility.

        Returns:
        - A tuple containing the training and validation datasets, each as a tuple of (Input_data, target_data).
        """
        num_elements = tf.shape(self.Input_data)[0].numpy()
        indices = self.batch_shuffle(num_elements, seed, BATCH_SIZE=BATCH_SIZE)
        
        # Determine the split point
        split_point = int(num_elements * self.split_ratio)
        
        # Split the indices into training and validation indices
        train_indices = indices[:split_point]
        val_indices = indices[split_point:]

        # Create the training and validation datasets
        return ( tf.gather(self.Input_data, train_indices), tf.gather(self.Target_data, train_indices), 
                tf.gather(self.Input_data, val_indices), tf.gather(self.Target_data, val_indices) )

            
    class MyModel(tf.keras.Model):
        def __init__(self, dict_size, embedding_weights, rnn_units, prediction_length):
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
            self.attention = tf.keras.layers.Attention()
            #self.conv1d = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')
            #self.pooling = tf.keras.layers.GlobalMaxPooling1D()
            self.dense = tf.keras.layers.Dense(dict_size*prediction_length, activation='softmax')
            self.reshape = tf.keras.layers.Reshape((prediction_length, dict_size))
            
        @tf.function
        def call(self, inputs, states=None, return_state=False, training=False):
            embedding = self.embedding(inputs, training=True)
            
            if states is None:
                states = self.gru.get_initial_state(embedding)
            gru, states = self.gru(embedding, initial_state=states, training=training)
            flatten = self.flatten(gru)
            attention = self.attention([flatten, flatten]) # to make sure silences are not over represented
            #x = tf.expand_dims(attention, axis=2) 
            #conv1d = self.conv1d(x)
            #pooling = self.pooling(conv1d)
            outputs  = self.dense(attention, training=training)
            reshaped_outputs = self.reshape(outputs)
            
            if return_state:
                return reshaped_outputs, states
            else:
                return reshaped_outputs
            
            
    def build_model(self,  rnn_units = 128):
        def kld_loss(y_true, y_pred):
            return tf.keras.losses.KLDivergence()(y_true, y_pred)
                
        self.model = self.MyModel(self.dict_size, self.embedding_weights, rnn_units = rnn_units, prediction_length=self.prediction_length)
        #self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        #self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        self.model.compile(optimizer='adam', loss=kld_loss)

        
        
    def plot_learning_curve(self, history):
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
        
    def train_model(self,  BATCH_SIZE=128, EPOCHS=20):
        self.BATCH_SIZE = BATCH_SIZE  # Training batch size
        self.EPOCHS = EPOCHS  # Number of training epochs
        
        self.history = self.model.fit(self.Input_data, self.Target_data, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=self.validation_data)
        self.plot_learning_curve(self.history)
        
        
    
    def random_predict(self, prediction, temperature=.99):
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
        
    def compile_model(self,  input_num=8, temperature=.5, max_len=60):
        self.input_num=input_num
        self.temperature=temperature
        self.max_len=max_len
        
    
    def __call__(self, seed):
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
        temperature=self.temperature
        states=None
        # define the input matrix
        input_tokens = self.BasslineLibrary.Data[seed]
        input_tokens = tf.boolean_mask(input_tokens, input_tokens!=0)[:self.input_num]
    
        # iteratively predict the next tokens till the bar if filled
        iter_num = BasslineLibrary.Data[seed].shape[0] - self.input_num -1
        #for i in range(0, iter_num, prediction_length):
        getOut=True
        i=0
        while getOut:
            prediction_mat, states = self.model(input_tokens[None,i:self.input_num+i], states=states, return_state=True)
            prediction_mat = tf.squeeze(prediction_mat)
            for j in range(self.prediction_length):
                pred1 = tf.gather(prediction_mat, j)
                prediction = self.random_predict(pred1, self.temperature)
                input_tokens = np.concatenate([input_tokens,prediction[0]])
                
                temperature=temperature*.999
                i+=1
                
                if prediction[0]==0 or i>iter_num: 
                    getOut=False
                    break
            
        print("\nPredicted Bassline")
        self.BasslineLibrary.print_detokenize(np.append(input_tokens, [0]))
        self.temperature_evantual=temperature

        



#%% Parameters
# Data 
N = 6000                                                                       # amount of data
window_size = 12                                                               # the input token length
prediction_length = 6                                                          # the number of tokens to predict per step 
step = 3                                                                       # number of tokens to jump in training 
split_ratio=.75                                                                # training / validation split

autofillBassTab = AutofillBassTab(N, window_size, prediction_length, step, split_ratio)


#%% loading data
# Load the list from the Pickle file
with open('BasslineLibrary.pickle', 'rb') as f:
    BasslineLibrary = pickle.load(f)

# Load the pre-trained embeddings
embedding_weights = np.load('Embeddings.npy')

autofillBassTab.load_dataset(embedding_weights, BasslineLibrary)


#%% build and run model
autofillBassTab.build_model(rnn_units=128)
autofillBassTab.train_model(BATCH_SIZE=128, EPOCHS=8) 

autofillBassTab.model.summary()


#%% predict
seed=50
autofillBassTab.compile_model(temperature=.8, input_num=window_size)

print("\nReference Bassline - Song: "+BasslineLibrary.names[seed])
BasslineLibrary.print_detokenize(BasslineLibrary.Data[seed])

print("\nInput Bassline - Song: "+BasslineLibrary.names[seed])
BasslineLibrary.print_detokenize(BasslineLibrary.Data[seed][:window_size])

for _ in range(3):
    autofillBassTab(seed)
    
'''
#%% save
# Save the entire model
autofillBassTab.model.save('autofill_model')


#%% reload
autofillBassTab_reloaded = AutofillBassTab(N, window_size, prediction_length, step, split_ratio, 'autofill_model')
autofillBassTab_reloaded.load_dataset(embedding_weights, BasslineLibrary)


#%% predict
seed=50
autofillBassTab_reloaded.compile_model(temperature=.9, input_num=window_size)

print("\nReference Bassline - Song: "+BasslineLibrary.names[seed])
BasslineLibrary.print_detokenize(BasslineLibrary.Data[seed])

print("\nInput Bassline - Song: "+BasslineLibrary.names[seed])
BasslineLibrary.print_detokenize(BasslineLibrary.Data[seed][:window_size])

for _ in range(3):
    autofillBassTab_reloaded(seed)
'''