# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 18:10:02 2023

@author: Mels
"""
import ast

class BassTokens:
    def __init__(self, G,D,A,E, name="", artist="",genre=""):
        '''
        Class to represent a tablature for bass guitar in tokenized form.
        
        Parameters
        ----------
        G : list
            A list containing strings representing the notes on the G string of the bass guitar.
        D : list
            A list containing strings representing the notes on the D string of the bass guitar.
        A : list
            A list containing strings representing the notes on the A string of the bass guitar.
        E : list
            A list containing strings representing the notes on the E string of the bass guitar.
        name : str, optional
            The name of the song. Defaults to an empty string.
        artist : str, optional
            The name of the artist. Defaults to an empty string.
        genre : str, optional
            The genre of the song. Defaults to an empty string.
    
        Raises
        ------
        ValueError
            If the length of `G`, `D`, `A`, and `E` lists are not equal.
    
        Returns
        -------
        None.
        
        '''
        self.name=name if name is not None else ""
        self.artist=artist if artist is not None else ""
        self.genre=genre if genre is not None else ""
        self.generate_dicts()
        
        if len(G)!=len(D) and len(G)!=len(A) and len(G)!=len(E): 
            raise ValueError("Objects [G,D,A,E] do not have equal length!")
        self.tokens = self.tokenize_bar(G, D, A, E )
            
    
    def tokenize_bar(self, G, D, A, E):
        """
        Tokenizes a bar of guitar tablature and returns the count of each fret for each string.
    
        Parameters
        ----------
        G : str
            The string representing the notes on the 3rd string of the guitar.
        D : str
            The string representing the notes on the 2nd string of the guitar.
        A : str
            The string representing the notes on the 4th string of the guitar.
        E : str
            The string representing the notes on the 1st string of the guitar.
    
        Returns
        -------
        bar_counts : list
            A list containing the count of each fret for each string in the given order [E, A, D, G].
    
        """
        strings = [E, A, D, G]
        bar_counts = []
        bar_len = len(G)
        
        i=0
        while i < bar_len:
            count = [0,0,0,0,0]
            doubledigits = False
            containsspecial = False
            ghostnote = False
            #skipsave = False
            
            for j in range(4):
                # double digits smaller than 20
                if (i < bar_len-1 and strings[j][i] in "0123456789" and strings[j][i+1] in "0123456789" 
                    and int(strings[j][i]+strings[j][i+1])<=20 and strings[j][i] not in '0'):
                    doubledigits=True
                    count[j] = self.dict_frets[strings[j][i]+strings[j][i+1]]
                    
                    # special characters
                    if i < bar_len-2 and strings[j][i+2] in self.speciallist:
                        containsspecial = True
                                                
                        # we will correct the slides and pulloffs to be in the correct direction. Therefore we need to look
                        # at which number comes after it, and if it is bigger or smaller
                        if ( i < bar_len-4 and strings[j][i+3] in "0123456789" and strings[j][i+4] in "0123456789"
                            and int(strings[j][i+3]+strings[j][i+4])<=20):
                            count[4]=self.correct_special(strings[j][i]+strings[j][i+1], strings[j][i+2],
                                                          strings[j][i+3]+strings[j][i+4])
                        elif i < bar_len-3 and strings[j][i+3] in "0123456789":
                            count[4]=self.correct_special(strings[j][i]+strings[j][i+1], strings[j][i+2], strings[j][i+3])
                        else:
                            count[4] = self.dict_special[strings[j][i+2]]
                        
                # single digits
                elif strings[j][i] in "0123456789":
                    count[j] = self.dict_frets[strings[j][i]]
                    
                    # special characters
                    if i < bar_len-1 and strings[j][i+1] in self.speciallist:
                        containsspecial = True
                        
                        # we will correct the slides and pulloffs to be in the correct direction. Therefore we need to look
                        # at which number comes after it, and if it is bigger or smaller
                        if ( i < bar_len-3 and strings[j][i+2] in "0123456789" and strings[j][i+3] in "0123456789"
                            and int(strings[j][i+2]+strings[j][i+3])<=20):
                            count[4]=self.correct_special(strings[j][i], strings[j][i+1],
                                                          strings[j][i+2]+strings[j][i+3])
                        elif i < bar_len-2 and strings[j][i+2] in "0123456789":
                            count[4]=self.correct_special(strings[j][i], strings[j][i+1], strings[j][i+2])
                        else:
                            count[4] = self.dict_special[strings[j][i+1]]
                    
                # dead note
                elif strings[j][i] in "x":
                    count[j] = self.dict_frets["x"]
                    
                # ghost note
                elif strings[j][i] in "(":
                    count[4] = self.dict_special["g"]
                    ghostnote = True
                    if ( strings[j][i+1] in "0123456789" and strings[j][i+2] in "0123456789"
                        and int(strings[j][i+1]+strings[j][i+2])<=20):
                        doubledigits=True
                        count[j] = self.dict_frets[strings[j][i+1]+strings[j][i+2]]                       
                    elif strings[j][i+1] in "0123456789":
                        count[j] = self.dict_frets[strings[j][i+1]]
                        
                # bars 
                elif strings[j][i] in "|":
                    #skipsave=True # do not save the bars
                    count[j] = self.dict_frets["|"]
                    
            #if not skipsave:
            bar_counts.append(str(count))
            
            # double digits and special character means an extra count needs to be skipped
            i+=1
            if doubledigits: i+=1
            if containsspecial: i+=1
            if ghostnote: i+=2     
            
        return bar_counts
    
    
    def print_detokenize(self):
        # print the tab
        G,D,A,E = self.detokenize()
        #for i in range(len(A)):
        print("G"+G)
        print("D"+D)
        print("A"+A)
        print("E"+E,'\n')   
        
        
    def detokenize(self):
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
        
        for count_str in self.tokens:
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
    
    
    def correct_special(self, note_1, special, note_2):
        '''
        Returns the corrected special character based on the given notes and special character.
    
        Parameters
        ----------
        note_1 : str
            The first note in the pair.
        special : str
            The special character to be corrected.
        note_2 : str
            The second note in the pair.
    
        Returns
        -------
        int
            The corrected special character, represented as an integer according to the self.dict_special dictionary.
    
        Raises
        ------
        None
    
        Notes
        -----
        - If the special character is one of '/', '\', 's', or 'S', then the function will correct it based on the values of note_1 and note_2.
        - If the special character is one of 'h', 'H', 'p', or 'P', then the function will correct it based on the relative position of note_1 and note_2.
        - If the special character is not one of the above, the function returns 0.
    
        '''
        if special in '/\\sS':
            if note_1.isdigit() and note_2.isdigit():
                if int(note_1) > int(note_2):
                    return self.dict_special['s']
                else:
                    return self.dict_special['/']
        elif special in 'hHpP':
            if int(note_1) > int(note_2):
                return self.dict_special['p']
            else:
                return self.dict_special['h']                    
        else:
            return 0
        
        
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