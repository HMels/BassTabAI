# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:57:51 2023

@author: Mels
"""

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

def skipgram(Tokens, vocab_size, embedding_size, window_size=4, learning_rate=0.01, epochs=10, batch_size=128):
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

    Returns:
        A numpy array of shape (vocab_size, embedding_size) containing the word embeddings.
    """                
    # Define the model and loss function
    model = SkipGramModel(vocab_size, embedding_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Create training data as pairs for the Skip-Gram Word2Vec model. 
    # Specifically, it generates pairs of target words and their corresponding
    # context words based on a sliding window approach.
    target_words = []
    context_words = []
    step=0
    for Token in Tokens:
        if step%10==0: print("Creating Embedding - step:",str(step)+'/'+str(len(Tokens)))
        step+=1
        
        # delete the rows with only zeros or with a one in the first column
        flattened = tf.boolean_mask(Token.tokens, tf.math.reduce_sum(Token.tokens,axis=1)!= 0)
        flattened = tf.boolean_mask(flattened, flattened[:, 0] != 1)
        for i, target_token in enumerate(flattened):

            # loop over the special notes and add them
            notes_i = Token.note2index(target_token)
            notes_special = Token.special2index(target_token)
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
                    notes_j = Token.note2index(flattened[j,:])
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
