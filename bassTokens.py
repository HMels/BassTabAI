# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:30:20 2023

@author: Mels
"""
#import tensorflow as tf

class BassTokens:
    def __init__(self, G,D,A,E, name="", artist="",genre=""):
        self.name=name if name is not None else ""
        self.artist=artist if artist is not None else ""
        self.genre=genre if genre is not None else ""
        self.generate_dicts()
        
        if len(G)!=len(D) and len(G)!=len(A) and len(G)!=len(E): raise ValueError("Objects [G,D,A,E] do not have equal length!")
        self.token = self.tokenize_bar(G, D, A, E )
            
    
    def tokenize_bar(self, G, D, A, E):
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
            bar_counts.append(count)
            
            # double digits and special character means an extra count needs to be skipped
            i+=1
            if doubledigits: i+=1
            if containsspecial: i+=1
            if ghostnote: i+=2     
            
        return bar_counts
    
    
    def print_detokenize(self):
        G,D,A,E = self.detokenize()
        for i in range(len(A)):
            print("G"+G[i])
            print("D"+D[i])
            print("A"+A[i])
            print("E"+E[i],'\n')   
        
        
    def detokenize(self):
        G = ""
        D = ""
        A = ""
        E = ""
        
        for count in self.token:
            if ( len(self.invdict_frets[count[0]])==2 or len(self.invdict_frets[count[1]])==2
                or len(self.invdict_frets[count[2]])==2 or len(self.invdict_frets[count[3]])==2): dash="--"
            else: dash="-"
            
            # contains special characters
            if count[4]!=0:
                if count[0]!=0: E += self.invdict_frets[count[0]]+self.invdict_special[count[4]]
                else: E += dash+'-'
                if count[1]!=0: A += self.invdict_frets[count[1]]+self.invdict_special[count[4]]
                else: A += dash+'-'
                if count[2]!=0: D += self.invdict_frets[count[2]]+self.invdict_special[count[4]]
                else: D += dash+'-'
                if count[3]!=0: G += self.invdict_frets[count[3]]+self.invdict_special[count[4]]
                else: G += dash+'-'
                
            # does not contain special characters
            else:
                if count[0]!=0: E += self.invdict_frets[count[0]]
                else: E += dash
                if count[1]!=0: A += self.invdict_frets[count[1]]
                else: A += dash
                if count[2]!=0: D += self.invdict_frets[count[2]]
                else: D += dash
                if count[3]!=0: G += self.invdict_frets[count[3]]
                else: G += dash
                    
        return G, D, A, E
    
    
    def correct_special(self, note_1, special, note_2):
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
            8:'g'  # grace note
            }
