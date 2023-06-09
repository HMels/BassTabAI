# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:57:51 2023

@author: Mels
"""

import pickle
import numpy as np
import tensorflow as tf

class SkipGramModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size):
        super(SkipGramModel, self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_size, input_length=1, name='embedding')
        self.dense_layer = tf.keras.layers.Dense(vocab_size, activation='softmax', name='dense')

    def call(self, target_word):
        embedding = self.embedding_layer(target_word)
        output = self.dense_layer(embedding)
        return output


def skipgram(Tokens, vocab_size, embedding_size, window_size=6, learning_rate=0.01, epochs=3, batch_size=128, batch_stop=10):
    """
    Trains a Skip-gram Word2Vec model on a list of tensors.

    Parameters:
        Tokens (bassTokens list): A list of bassTokens. 
        vocab_size (int): The size of the vocabulary to use.
        embedding_size (int): The size of the embedding vectors.
        window_size (int): The size of the window to use for the notes (default 4).
        learning_rate (float): The learning rate to use for optimization (default 0.01).
        epochs (int): The number of epochs to train for (default 10).
        batch_size (int): The batch size to use for training (default 128).
        batch_limit (int): The number of tokens to process per batch (default 5000).

    Returns:
        A numpy array of shape (vocab_size, embedding_size) containing the word embeddings.
    """                
    # Define the model and loss function
    global Embeddings_weights
    global model
    model = SkipGramModel(vocab_size, embedding_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Create training data as pairs for the Skip-Gram Word2Vec model. 
    # Specifically, it generates pairs of target words and their corresponding
    # context words based on a sliding window approach.
    target_notes = []
    context_notes = []
    step_batch=0
    #step=0
    #tokenshuffle = np.arange(len(Tokens))
    #np.random.shuffle(tokenshuffle)
    for i in range(0, len(Tokens), batch_size):
        if step_batch==batch_stop: break
        print("Creating Embedding - Batch: "+str(step_batch)+"/"+str(min(np.round(len(Tokens)/batch_size),batch_stop)))
        step_batch+=1
        
        # take the random shuffled batches out
        batch_tokens = Tokens[i:min(i+batch_size, len(Tokens)-1)] #tokenshuffle[i:min(i+batch_size, len(tokenshuffle)-1)] ]
        for Token in batch_tokens:
            #if step%100==0: print("Creating Embedding - step:",str(step)+'/'+str(len(Tokens)))
            #step+=1
            
            # delete the rows with only zeros or with a one in the first column
            flattened = tf.boolean_mask(Token, Token==0)
            flattened = tf.boolean_mask(Token, Token==1)
            
            # create a list of context notes windowed over the surrounding notes
            for i, target_token in enumerate(flattened):
                start = max(0, i - window_size)
                end = min(len(flattened), i + window_size + 1)
                for j in range(start, end):
                    if j != i:
                        target_notes.append(flattened[i])
                        context_notes.append(flattened[j])
                        
        # Preparing the dataset for training the Word2Vec model by organizing the training 
        # examples into batches and randomizing the order in which they are presented to the model.
        dataset = tf.data.Dataset.from_tensor_slices((target_notes, context_notes))
        dataset = dataset.shuffle(buffer_size=len(target_notes)).batch(batch_size)
    
        # Train the model
        for epoch in range(epochs):
            epoch_loss = 0
            for target_batch, context_batch in dataset:
                with tf.GradientTape() as tape:
                    output = model(target_batch)
                    loss = loss_fn(context_batch, output)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                epoch_loss += loss
                
            # Get the word embeddings
            Embeddings_weights = model.get_layer('embedding').get_weights()[0]
            print('Epoch:', epoch, 'Loss:', epoch_loss)
    
        # Clear the lists for the next batch
        target_notes.clear()
        context_notes.clear()



#%% create embeddings
# Load the list back from the Pickle file
with open('BasslineLibrary.pickle', 'rb') as f:
    BasslineLibrary = pickle.load(f)
    
embedding_size = (21+3)*4 + 9 # 21 frets (incl zeroth), 3 special notes (dead, none and bar) and 9 special moves
skipgram(BasslineLibrary.Data, vocab_size=len(BasslineLibrary.library), embedding_size=embedding_size)

np.save('Embeddings.npy', Embeddings_weights)
