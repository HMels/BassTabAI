## BassTabAI
 
Right now the architecture of the program is the following:

1. Loading the bass tabs (loadBassTab.py)

	1.1 The file loadBassTab.py uses the function load_bassTab() to load bass tabs from the internet. It loads it into the BassTab class to get it into the right format (reading the basslines of the internet is a bit messy). It also uses the function parse_tab_line() from bassTab.py to load basslines till the end of the line.
	
	1.2 It will then create a BassTokens, which is a class from the file with the same name. This class is able to correctly split the basstab into tokens, by checking the surrounding notes. 
	It is able to store them into the next format [E,A,D,G,S] which are the 4 strings and then the special move (hammer ons, slides, pull-offs, bends, and more). This is done via a dictionary.
		
	1.3 Then it adds the tokens to an overarching class called BasslineLibrary. This class generates a library of all tokens it gets as input, and will store the basslines as references to elemens in that library. It also stores other information such as song name, artist and genre. The first 5 library items look as follows:
	
			- [23, 23, 23, 23, 0]; 0,
			- [0, 0, 0, 0, 0]; 1,
			- [0, 8, 0, 0, 0]; 2,
			- [0, 11, 0, 0, 0]; 3,
			- [0, 6, 0, 0, 0]; 4,
			- [0, 4, 0, 0, 0]; 5, .....
	
	1.4 It then saves the library as BasslineLibrary.pickle

2. Next, Tab2Vec.py will crerate an Embeddings using the skip gram model found in Tab2Vec.

	2.1 The embeddings will be saved as Embeddings.npy

3. Finally, the model autofillBassTab will be created in autofillBassTab.py. The neural network is based on a model by https://jaketae.github.io/study/auto-complete/, with some extra insights from https://www.tensorflow.org/text/tutorials/text_generation. 

	3.1 First the input data is split into input_tokens and targets. We let the model take in a certain number of previous tokens after which it will predict the next one. This means input_tokens will have size (Ndata x 12) for window size 12 and targets is only an array of (Ndata x 1)
	
	3.2 Next we split the input data into a training and a validation dataset, with 80% of the dataset being used for training, and the rest for validation.
	
	3.3 Training: Right now we have decided the neural network architecture to be as given below. We have chosen to use a GRU (Gated Recurrent Unit) as it can remember states. This will come in handy as the model will then be able to memorise its state during previous predictions and thus be more coherent when predicting multiple tokens in sequence.
		
		_________________________________________________________________
		 Layer (type)                Output Shape              Param #   
		=================================================================
		 embedding (Embedding)       multiple                  147210    
																		 
		 gru    (GRU)                multiple                  90240     
																		 
		 flatten    (Flatten)        multiple                  0         
																		 
		 attention    (Attention)    multiple                  0         
																		 
		 dense    (Dense)            multiple                  5920524   
																		 
		 reshape    (Reshape)        multiple                  0         
																		 
		=================================================================
		Total params: 6,157,974
		Trainable params: 6,157,974
		Non-trainable params: 0
		_________________________________________________________________
		
	During training we discovered that using a pretrained embedding does not work well. Therefore we also train the embedding with all the other layers. The dense layer is the output layers. 
		
	We have also added an attention layer as the data contains a lot of empty notes, and we do not want the program to over-predict those.	

	3.4 After training, we let the model take in part of the song, and make it predict the next token step for step. This next token is then added to the total tableture, and will thereafter be used in predicting the following tokens. States of the GRU layers are also passed to the next iteration. This iterative process is repeated till the predicted tab is as large as the original. 
		In predicting, we have used a temperature input that will generate some randomness in the output, otherwise its prediction will be boring and often does not contain a lot of notes (silences are the most common notes). We use a temperature of 0.99 that is decreased per iteration by multiplying it with 0.997 to make sure it does not become too random in the end. 
	
	For example, we have used the song  Money from Pink Floyd:

		G|------------------------------------13----------11-------------------|
		D|-----4---3---2--------0-0-0---------------------------------4--------|
		A|-----------------2--2--------4-----------9---------------------------|
		E|-2-------------------------------------------------------------------| 


		Predicted Bassline
		G|-------------------------------------------------------------------|
		D|-----4---3-----------4---------------------------------------------|
		A|-------------------------------------------------------------------|
		E|-2---------------2--------------5----------------------------------| 


		Predicted Bassline
		G|-------------------------------------------------------------------|
		D|-----4---3---2---------------2-------------------------------------|
		A|----------------------3--3--3-----------3-----------------------2--|
		E|-2----------------------------------------------3----------3---3---| 