# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 18:50:58 2023

The total created library of the basslines

@author: Mels
"""
import tensorflow as tf
import ast


class BasslineLibrary:
    def __init__(self):
        """
        Initializes a new BasslineLibrary instance with an empty dictionary and a counter set to 0.
        """
        self.library = {'[23, 23, 23, 23, 0]': 0,
         '[0, 0, 0, 0, 0]': 1,}
        self.counter = 1
        self.Data = []
        self.names = []
        self.artists = []
        self.genres = []
        self.generate_dicts()
        
        
    def add_tokens(self, Token):
        '''
        Add an input to our database via the library. Will add vocabulary to the library if needed

        Parameters
        ----------
        Token : BassToken class
            Class containing the tokens and misc info of the bassline.

        Returns
        -------
        None.

        '''
        self.names.append(Token.name)
        self.artists.append(Token.artist)
        self.genres.append(Token.genre)
        
        translated_tokens = []
        for token in Token.tokens:
            lib_token = self.add_count(token)
            translated_tokens.append(lib_token)
        self.Data.append(tf.Variable(translated_tokens))


    def add_count(self, count):
        """
        Adds a new count to the library if it doesn't exist yet and returns its assigned number.
        If the count already exists, it returns its assigned number.
        
        Parameters:
        count (str): The count to add to the library.
        
        Returns:
        int: The assigned number of the count.
        """
        if count not in self.library:
            self.counter += 1
            self.library[count] = self.counter
        return self.library[count]


    def get_library(self):
        """
        Returns the current state of the library.
        
        Returns:
        dict: The library of counts with their assigned numbers.
        """
        return self.library


    def translate_bassline(self, bassline):
        """
        Translates a list of counts to a list of their assigned numbers in the library.
        
        Parameters:
        bassline (list[str]): The list of counts to translate.
        
        Returns:
        list[int]: The translated bassline with numbers corresponding to the counts in the library.
        """
        translated_bassline = []
        for count in bassline:
            translated_bassline.append(self.add_count(count))
        return translated_bassline


    def get_counts(self, translated_bassline):
        """
        Returns the counts corresponding to a translated bassline.
        
        Parameters:
        translated_bassline (list[int]): The list of numbers to convert back to counts.
        
        Returns:
        list[str]: The counts corresponding to the numbers in the translated bassline.
        """
        counts = []
        for number in translated_bassline:
            for count, num in self.library.items():
                if num == number:
                    counts.append(count)
                    break
        return counts


    def get_translated_bassline(self, counts):
        """
        Translates a list of counts to a list of their assigned numbers in the library.
        
        Parameters:
        counts (list[str]): The list of counts to translate.
        
        Returns:
        list[int]: The translated bassline with numbers corresponding to the counts in the library.
        """
        translated_bassline = []
        for count in counts:
            translated_bassline.append(self.add_count(count))
        return translated_bassline
    
    
    
    ################# TO RETURN THE TAB #######################################
    def print_detokenize(self, translated_bassline):
        # print the tab
        G,D,A,E = self.detokenize(translated_bassline)
        #for i in range(len(A)):
        if G[0]=="|": print("G"+G)
        else: print("G|"+G) 
        if G[0]=="|": print("D"+D)
        else: print("D|"+D)
        if G[0]=="|": print("A"+A)
        else: print("A|"+A)
        if G[0]=="|": print("E"+E,'\n') 
        else: print("E|"+E,'\n')  
        
        
    def detokenize(self, translated_bassline):
        """
        Convert the tokenized fretboard back into a string representation.
    
        Returns
        -------
        G : str
            A string representing the sixth string of the fretboard.
        D : str
            A string representing the fourth string of the fretboard.
        A : str
            A string representing the third string of the fretboard.
        E : str
            A string representing the first string of the fretboard.
        """
        G = ""
        D = ""
        A = ""
        E = ""
        tokens = self.get_counts(translated_bassline)
        
        for count_str in tokens:
            count = ast.literal_eval(count_str) # transform the string to a list again
            
            if ( len(self.invdict_frets[count[0]])==2 or len(self.invdict_frets[count[1]])==2
                or len(self.invdict_frets[count[2]])==2 or len(self.invdict_frets[count[3]])==2): dash="--"
            else: dash="-"
            G1,D1,A1,E1='','','',''
            # contains special characters
            if count[4]!=0:
                if count[0]!=0: E1 += self.invdict_frets[count[0]]+self.invdict_special[count[4]]
                else: E1 += dash+'-'
                if count[1]!=0: A1 += self.invdict_frets[count[1]]+self.invdict_special[count[4]]
                else: A1 += dash+'-'
                if count[2]!=0: D1 += self.invdict_frets[count[2]]+self.invdict_special[count[4]]
                else: D1 += dash+'-'
                if count[3]!=0: G1 += self.invdict_frets[count[3]]+self.invdict_special[count[4]]
                else: G1 += dash+'-'
                
            # does not contain special characters
            else:
                if count[0]!=0: E1 += self.invdict_frets[count[0]]
                else: E1 += dash
                if count[1]!=0: A1 += self.invdict_frets[count[1]]
                else: A1 += dash
                if count[2]!=0: D1 += self.invdict_frets[count[2]]
                else: D1 += dash
                if count[3]!=0: G1 += self.invdict_frets[count[3]]
                else: G1 += dash
                
            # align all numbers with dashes
            maxlen = max(len(G1),len(D1),len(A1),len(E1))
            G1+= "-"*(maxlen-len(G1))
            D1+= "-"*(maxlen-len(D1))
            A1+= "-"*(maxlen-len(A1))
            E1+= "-"*(maxlen-len(E1))
            
            G+=G1
            D+=D1
            A+=A1
            E+=E1
                    
        G = G.replace('||','|')
        D = D.replace('||','|')
        A = A.replace('||','|')
        E = E.replace('||','|')
        return G, D, A, E
    
    
    def generate_dicts(self):
        '''
        Creates dictionaries and lists to map between tokens and their 
        corresponding frets, special characters, and dead notes.
    
        Returns
        -------
        None.
        '''
        self.dict_frets = {
            '-': 0,
            '0': 1,
            '1': 2,
            '2': 3,
            '3': 4,
            '4': 5,
            '5': 6,
            '6': 7,
            '7': 8,
            '8': 9,
            '9': 10,
            '10': 11,
            '11': 12,
            '12': 13,
            '13': 14,
            '14': 15,
            '15': 16,
            '16': 17,
            '17': 18,
            '18': 19,
            '19': 20,
            '20': 21,
            'x': 22,
            '|': 23
            }
        
        self.invdict_frets = {
            0: '-', # no note
            1: '0',
            2: '1',
            3: '2',
            4: '3',
            5: '4',
            6: '5',
            7: '6',
            8: '7',
            9: '8',
            10: '9',
            11: '10',
            12: '11',
            13: '12',
            14: '13',
            15: '14',
            16: '15',
            17: '16',
            18: '17',
            19: '18',
            20: '19',
            21: '20',
            22: 'x', # dead note
            23: '|', # end of bar
            }
        
        self.speciallist = "/\\sShHpPbB^~*."
        
        self.dict_special= {
            '\\':1,
            '/':2,
            's':1,
            'S':1,
            'h':3,
            'H':3,
            'P':4,
            'p':4,
            'b':5,
            'B':5,
            '^':5,
            '~':6,
            '*':7,
            '.':7,
            'g':8
            }
        
        self.invdict_special = {
            0:'',
            1:'s', # slide down
            2:'/', # slide up
            3:'h', # hammer-on
            4:'p', # pull-off
            5:'b', # bend
            6:'~', # let ring
            7:'*', # Staccato
            8:'g'  # ghots note
            }
