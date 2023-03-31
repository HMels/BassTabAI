# BassTabAI
 
Right now the architecture of the program is the following:

1. Loading the bass tabs (loadBassTab.py)
	1.1 The file loadBassTab.py uses the function load_bassTab() to load bass tabs from the internet. It loads it into the BassTab class to get it into the right format (reading the basslines of the internet is a bit messy). It also uses the function parse_tab_line() from bassTab.py to load basslines till the end of the line.
	1.2 It will then create a BassTokens, which is a class from the file with the same name. This class is able to correctly split the basstab into tokens, by checking the surrounding notes. 
		It is able to store them into the next format [E,A,D,G,S] which are the 4 strings and then the special move (hammer ons, slides, pull-offs, bends, and more). This is done via a dictionary.
	1.3 Then it adds the tokens to an overarching class called BasslineLibrary. This class generates a library of all tokens it gets as input, and will store the basslines as references to elemens in that library. It also stores misc information such as song name, artist and genre.
	1.4 It then saves the library as BasslineLibrary.pickle

2. Next, loadBassTab.py will crerate an Embeddings using the skip gram model found in Tab2Vec.
	2.1 The embeddings will be saved as Embeddings.npy

3. Finally, the model PredictBassTab will be created in predictBassTab.py. The architecture can be found in that class.
	3.1 The model is trained using train_model() in the same file. 
	3.2 The model will then be saved under the PredictBassTab tf.keras folder
	3.3 The model will then be tested on some example data