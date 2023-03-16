# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:30:20 2023

@author: Mels
"""

class Embeddings:
    def __init__(self, bassTabs):
        self.name=bassTabs.name if bassTabs.name is not None else ""
        self.artist=bassTabs.artist if bassTabs.artist is not None else ""
        self.generate_dicts()
        self.token = []
        for i in range(len(bassTabs.A)):
            self.token.append( self.tokenize_bar(bassTabs.G[i], bassTabs.D[i], bassTabs.A[i], bassTabs.E[i] ) )
            
    
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
            
            for j in range(4):
                # double digits smaller than 20
                if (i < bar_len-1 and strings[j][i] in "0123456789" and strings[j][i+1] in "0123456789" 
                    and int(strings[j][i]+strings[j][i+1])<=20 and strings[j][i] not in '0'):
                    doubledigits=True
                    count[j] = self.dict_frets[strings[j][i]+strings[j][i+1]]
                    
                    # special characters
                    if i < bar_len-2 and strings[j][i+2] in "/\\sShHpPbB^~*":
                        containsspecial = True
                        count[4] = self.dict_special[strings[j][i+2]]
                        
                # single digits
                elif strings[j][i] in "0123456789":
                    count[j] = self.dict_frets[strings[j][i]]
                    
                    # special characters
                    if i < bar_len-1 and strings[j][i+1] in "/\sShHpPbB^~*":
                        containsspecial = True
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
                    count[j] = self.dict_frets["|"]
                    
                
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
        G,D,A,E=[],[],[],[]
        for i in range(len(self.token)):
            G1,D1,A1,E1=self.detokenize_bar(self.token[i])
            G.append(G1)
            D.append(D1)
            A.append(A1)
            E.append(E1)
        return G,D,A,E
    
    def detokenize_bar(self, bar_counts):
        G = ""
        D = ""
        A = ""
        E = ""
        
        for count in bar_counts:
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
            0: '-',
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
            22: 'x',
            23: '|',
            }

        
        self.dict_special= {
            '\\':1,
            '/':1,
            's':1,
            'h':2,
            'H':2,
            'P':3,
            'p':3,
            'b':4,
            'B':4,
            '^':4,
            '~':5,
            '*':6,
            'g':7
            }
        
        self.invdict_special = {
            0:'',
            1:'s',
            2:'h',
            3:'p',
            4:'b',
            5:'~',
            6:'*',
            7:'g'
            }
