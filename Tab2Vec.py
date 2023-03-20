# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:57:51 2023

@author: Mels
"""

import tensorflow as tf
import numpy as np

class SkipGramModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size):
        super(SkipGramModel, self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_size, input_length=1, name='embedding')
        self.dense_layer = tf.keras.layers.Dense(vocab_size, activation='softmax', name='dense')

    def call(self, target_word):
        embedding = self.embedding_layer(target_word)
        output = self.dense_layer(embedding)
        return output

def skipgram(Tokens, vocab_size, embedding_size, window_size=16, learning_rate=0.01, epochs=10, batch_size=128):
    """
    Trains a Skip-gram Word2Vec model on a list of tensors.

    Parameters:
        Tokens (bassTokens list): A list of bassTokens. 
        vocab_size (int): The size of the vocabulary to use.
        embedding_size (int): The size of the embedding vectors.
        window_size (int): The size of the window to use for context words (default 2).
        learning_rate (float): The learning rate to use for optimization (default 0.01).
        epochs (int): The number of epochs to train for (default 10).
        batch_size (int): The batch size to use for training (default 128).

    Returns:
        A numpy array of shape (vocab_size, embedding_size) containing the word embeddings.
    """
    '''
    # Flatten the tensors and create a vocabulary
    tensors=[]
    for Token in Tokens:
        boolean_mask = tf.cast(Token.tokens, dtype=tf.bool)              
        no_zeros = tf.boolean_mask(Token.tokens, boolean_mask, axis=0)
        tensors.append(no_zeros)
    flattened = np.concatenate(tensors, axis=0)
    vocab, counts = np.unique(flattened, return_counts=True, axis=0)        
    #word2id = {w: i for i, w in enumerate(vocab)} # convert target words to their corresponding IDs
    #id2word = {i: w for i, w in enumerate(vocab)} # convert IDs to their corresponding target words

    # Define the model and loss function
    model = SkipGramModel(vocab_size, embedding_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Create training data as pairs for the Skip-Gram Word2Vec model. 
    # Specifically, it generates pairs of target words and their corresponding
    # context words based on a sliding window approach.
    target_words = []
    context_words = []
    for i, target_word in enumerate(flattened):
        start = max(0, i - window_size)
        end = min(len(flattened), i + window_size + 1)
        for j in range(start, end):
            if j != i:
                #target_words.append(word2id[target_word])
                #context_words.append(word2id[flattened[j]])
                target_words.append(np.argwhere(vocab==target_word))
                context_words.append(np.argwhere(vocab==flattened[j]))
    '''
                
    # Define the model and loss function
    model = SkipGramModel(vocab_size, embedding_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Create training data as pairs for the Skip-Gram Word2Vec model. 
    # Specifically, it generates pairs of target words and their corresponding
    # context words based on a sliding window approach.
    target_words = []
    context_words = []
    for Token in Tokens:
        Nnotes = Token.tokens.shape[1] - Token.Nspecial
        
        # delete the rows with only zeros or with a one in the first column
        flattened = tf.boolean_mask(Token.tokens, tf.math.reduce_sum(Token.tokens,axis=1)!= 0)
        flattened = tf.boolean_mask(flattened, flattened[:, 0] != 1)
        for i, target_token in enumerate(flattened):

            # loop over the special notes and add them
            notes_i = tf.where(target_token[:Nnotes])
            notes_special = tf.where(target_token[Nnotes:])+Nnotes
            if len(notes_special)!=0:
                for note_i in notes_i:
                    for special in notes_special:
                        target_words.append(note_i.numpy()[0])
                        context_words.append(special.numpy()[0])
                 
            # windowed over the surrounding notes
            start = max(0, i - window_size)
            end = min(len(flattened), i + window_size + 1)
            for j in range(start, end):
                if j != i:
                    notes_j = tf.where(flattened[j,:Nnotes])
                    for note_i in notes_i:
                        for note_j in notes_j:
                            target_words.append(note_i.numpy()[0])
                            context_words.append(note_j.numpy()[0])


    # Preparing the dataset for training the Word2Vec model by organizing the training 
    # examples into batches and randomizing the order in which they are presented to the model.
    dataset = tf.data.Dataset.from_tensor_slices((target_words, context_words))
    dataset = dataset.shuffle(buffer_size=len(target_words)).batch(batch_size)

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
        print('Epoch:', epoch, 'Loss:', epoch_loss)

    # Get the word embeddings
    embeddings = model.get_layer('embedding').get_weights()[0]
    return embeddings




'''
def generate_training_data(flattened, word2id, id2word, window_size, batch_size, embedding_size):
    # Split the flattened tensor into smaller chunks
    chunk_size = 10000  # Number of tokens in each chunk
    chunks = [flattened[i:i+chunk_size] for i in range(0, len(flattened), chunk_size)]

    # Generate training pairs for each chunk
    target_words = []
    context_words = []
    for chunk in chunks:
        chunk_target_words = []
        chunk_context_words = []
        for i, target_word in enumerate(chunk):
            start = max(0, i - window_size)
            end = min(len(chunk), i + window_size + 1)
            for j in range(start, end):
                if j != i:
                    chunk_target_words.append(word2id[target_word])
                    chunk_context_words.append(id2word[chunk[j]])
        target_words.extend(chunk_target_words)
        context_words.extend(chunk_context_words)

    # Create dataset and iterator
    dataset = tf.data.Dataset.from_tensor_slices((target_words, context_words))
    dataset = dataset.shuffle(buffer_size=len(target_words)).batch(batch_size)
    iterator = dataset.make_initializable_iterator()

    # Define embedding matrix
    embeddings = tf.Variable(tf.random_uniform([len(word2id), embedding_size], -1.0, 1.0))

    # Define inputs and outputs
    inputs = tf.placeholder(tf.int32, shape=[None])
    outputs = tf.placeholder(tf.int32, shape=[None, 1])

    # Define embeddings lookup and loss
    embed = tf.nn.embedding_lookup(embeddings, inputs)
    nce_weights = tf.Variable(tf.truncated_normal([len(word2id), embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([len(word2id)]))
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=outputs, inputs=embed, num_sampled=batch_size, num_classes=len(word2id)))

    # Define optimizer and training operation
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

    # Return iterator, inputs, outputs, loss, optimizer, and embeddings
    return iterator, inputs, outputs, loss, optimizer, embeddings
'''